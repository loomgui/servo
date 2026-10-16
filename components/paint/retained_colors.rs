/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use rustc_hash::FxHashMap;
use servo_base::Epoch;
use servo_base::id::{PipelineId, WebViewId};
use webrender::Transaction;
use webrender_api::{
    ColorF, DynamicProperties, PropertyBindingId, PropertyBindingKey, PropertyValue,
};

struct PipelineColors {
    generation: u64,
    epoch: Epoch,
    colors: FxHashMap<PropertyBindingId, PropertyValue<ColorF>>,
    caret_binding: Option<(PropertyBindingKey<ColorF>, ColorF)>,
    // WR overrides survive ordinary full DLs even when a caret is temporarily
    // absent. Only an actual animated-property reset removes inactive samples.
    caret_sample: Option<PropertyValue<ColorF>>,
}

/// Latest submitted values, including caret samples. Every reset reinstalls all
/// current pipelines. Generation protects keys reused by a full display list;
/// epoch ordering prevents late deltas from rewinding a retained scene.
#[derive(Default)]
pub(crate) struct RetainedColors {
    pipelines: FxHashMap<(WebViewId, PipelineId), PipelineColors>,
}

impl RetainedColors {
    pub(crate) fn install(
        &mut self,
        webview: WebViewId,
        pipeline: PipelineId,
        generation: u64,
        epoch: Epoch,
        colors: Vec<PropertyValue<ColorF>>,
        caret_binding: Option<(PropertyBindingKey<ColorF>, ColorF)>,
    ) {
        let caret_sample = self
            .pipelines
            .get(&(webview, pipeline))
            .and_then(|p| p.caret_sample);
        self.pipelines.insert(
            (webview, pipeline),
            PipelineColors {
                generation,
                epoch,
                colors: colors
                    .into_iter()
                    .map(|color| (color.key.id, color))
                    .collect(),
                caret_binding,
                caret_sample,
            },
        );
    }

    pub(crate) fn update(
        &mut self,
        webview: WebViewId,
        pipeline: PipelineId,
        generation: u64,
        epoch: Epoch,
        colors: &[PropertyValue<ColorF>],
    ) -> bool {
        let Some(retained) = self.pipelines.get_mut(&(webview, pipeline)) else {
            return false;
        };
        if generation != retained.generation
            || epoch < retained.epoch
            || colors
                .iter()
                .any(|color| !retained.colors.contains_key(&color.key.id))
        {
            return false;
        }
        for color in colors {
            retained.colors.insert(color.key.id, *color);
        }
        retained.epoch = epoch;
        true
    }

    pub(crate) fn caret_binding(
        &self,
        webview: WebViewId,
        pipeline: PipelineId,
    ) -> Option<(PropertyBindingKey<ColorF>, ColorF)> {
        self.pipelines
            .get(&(webview, pipeline))
            .and_then(|p| p.caret_binding)
    }

    pub(crate) fn sample_carets(&mut self, colors: Vec<PropertyValue<ColorF>>) {
        // Match the baseline global caret reset: inactive caret overrides are
        // removed, while every background property is reinstalled below.
        for pipeline in self.pipelines.values_mut() {
            pipeline.caret_sample = None;
        }
        for color in colors {
            for pipeline in self.pipelines.values_mut() {
                if pipeline
                    .caret_binding
                    .is_some_and(|(key, _)| key.id == color.key.id)
                {
                    pipeline.caret_sample = Some(color);
                    break;
                }
            }
        }
    }

    pub(crate) fn all(&self) -> Vec<PropertyValue<ColorF>> {
        self.pipelines
            .values()
            .flat_map(|pipeline| {
                pipeline
                    .colors
                    .values()
                    .copied()
                    .chain(pipeline.caret_sample)
            })
            .collect()
    }

    /// All production color resets use this order and this latest-value store.
    /// The pinned WR append API otherwise retains every historical value.
    pub(crate) fn reset_transaction(&self, transaction: &mut Transaction) {
        transaction.reset_dynamic_properties();
        transaction.append_dynamic_properties(DynamicProperties {
            transforms: Vec::new(),
            floats: Vec::new(),
            colors: self.all(),
        });
    }

    pub(crate) fn remove_pipeline(&mut self, webview: WebViewId, pipeline: PipelineId) {
        self.pipelines.remove(&(webview, pipeline));
    }

    pub(crate) fn remove_webview(&mut self, webview: WebViewId) {
        self.pipelines
            .retain(|(other_webview, _), _| *other_webview != webview);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use servo_base::id::{TEST_PIPELINE_ID, TEST_WEBVIEW_ID};
    use webrender_api::PropertyBindingKey;

    fn install(
        colors: &mut RetainedColors,
        webview: WebViewId,
        pipeline: PipelineId,
        values: Vec<PropertyValue<ColorF>>,
    ) {
        colors.install(webview, pipeline, 1, Epoch(1), values, None);
    }
    fn update(
        colors: &mut RetainedColors,
        webview: WebViewId,
        pipeline: PipelineId,
        values: &[PropertyValue<ColorF>],
    ) -> bool {
        colors.update(webview, pipeline, 1, Epoch(2), values)
    }

    fn color(id: u64, red: f32, alpha: f32) -> PropertyValue<ColorF> {
        PropertyBindingKey::new(id).with(ColorF::new(red, 0.0, 0.0, alpha))
    }

    #[test]
    fn caret_reset_preserves_latest_background_and_other_pipeline() {
        let mut colors = RetainedColors::default();
        let other = PipelineId {
            index: servo_base::id::Index::new(77).unwrap(),
            ..TEST_PIPELINE_ID
        };
        install(
            &mut colors,
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            vec![color(1, 1.0, 1.0)],
        );
        colors
            .pipelines
            .get_mut(&(TEST_WEBVIEW_ID, TEST_PIPELINE_ID))
            .unwrap()
            .caret_binding = Some((color(2, 1.0, 1.0).key, ColorF::WHITE));
        install(
            &mut colors,
            TEST_WEBVIEW_ID,
            other,
            vec![color(3, 0.5, 1.0)],
        );
        assert!(update(
            &mut colors,
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            &[color(1, 0.0, 1.0)]
        ));
        colors.sample_carets(vec![color(2, 1.0, 0.0)]);
        let all = colors.all();
        assert_eq!(all.len(), 3);
        assert_eq!(all.iter().find(|c| c.key.id.uid == 1).unwrap().value.r, 0.0);
        assert_eq!(all.iter().find(|c| c.key.id.uid == 2).unwrap().value.a, 0.0);
        assert_eq!(all.iter().find(|c| c.key.id.uid == 3).unwrap().value.r, 0.5);
    }

    #[test]
    fn full_display_list_replaces_keys_and_rejects_obsolete_batch_atomically() {
        let mut colors = RetainedColors::default();
        install(
            &mut colors,
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            vec![color(1, 1.0, 1.0)],
        );
        install(
            &mut colors,
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            vec![color(2, 0.5, 1.0)],
        );
        assert!(!update(
            &mut colors,
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            &[color(2, 0.0, 1.0), color(1, 0.0, 1.0)]
        ));
        assert_eq!(colors.all().len(), 1);
        assert_eq!(colors.all()[0].value.r, 0.5);
        colors.sample_carets(vec![color(1, 1.0, 0.0)]);
        assert_eq!(colors.all().len(), 1);
    }

    #[test]
    fn pipeline_and_webview_exit_retire_dynamic_values() {
        let mut colors = RetainedColors::default();
        install(
            &mut colors,
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            vec![color(1, 1.0, 1.0)],
        );
        colors.remove_pipeline(TEST_WEBVIEW_ID, TEST_PIPELINE_ID);
        assert!(colors.all().is_empty());
        assert!(!update(
            &mut colors,
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            &[color(1, 0.0, 1.0)]
        ));
        install(
            &mut colors,
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            vec![color(2, 1.0, 1.0)],
        );
        colors.remove_webview(TEST_WEBVIEW_ID);
        assert!(colors.all().is_empty());
    }

    #[test]
    fn off_phase_survives_full_list_and_retained_update_with_bounded_reset_order() {
        use webrender::FrameMsg;
        let mut values = RetainedColors::default();
        let caret = color(2, 1.0, 1.0);
        values.install(
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            1,
            Epoch(1),
            vec![color(1, 1.0, 1.0)],
            Some((caret.key, caret.value)),
        );
        // Before a timer sample the binding uses the DL's own default color.
        assert_eq!(values.all().len(), 1);
        values.sample_carets(vec![color(2, 0.0, 0.0)]);
        values.install(
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            2,
            Epoch(2),
            vec![color(1, 0.5, 1.0)],
            Some((caret.key, caret.value)),
        );
        for epoch in 3..103 {
            assert!(values.update(
                TEST_WEBVIEW_ID,
                TEST_PIPELINE_ID,
                2,
                Epoch(epoch),
                &[color(1, epoch as f32, 1.0)]
            ));
            let mut transaction = Transaction::new();
            values.reset_transaction(&mut transaction);
            let operations = transaction.get_frame_ops();
            assert_eq!(operations.len(), 2);
            assert!(matches!(operations[0], FrameMsg::ResetDynamicProperties));
            let FrameMsg::AppendDynamicProperties(properties) = &operations[1] else {
                panic!("reset must precede append");
            };
            assert_eq!(properties.colors.len(), 2);
            assert_eq!(
                properties
                    .colors
                    .iter()
                    .find(|v| v.key == caret.key)
                    .unwrap()
                    .value
                    .a,
                0.0
            );
            assert_eq!(
                properties
                    .colors
                    .iter()
                    .find(|v| v.key.id.uid == 1)
                    .unwrap()
                    .value
                    .r,
                epoch as f32
            );
        }
        // Ordinary full lists preserve the existing WR override even while the
        // caret is absent. A real animation reset then drops inactive samples.
        values.install(
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            3,
            Epoch(103),
            vec![color(1, 0.0, 1.0)],
            None,
        );
        assert_eq!(values.all().len(), 2);
        values.sample_carets(Vec::new());
        assert_eq!(values.all().len(), 1);
    }

    #[test]
    fn reused_key_rejects_old_generation_and_backward_epoch() {
        let mut colors = RetainedColors::default();
        colors.install(
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            1,
            Epoch(1),
            vec![color(1, 1.0, 1.0)],
            None,
        );
        // Actual full-list behavior reuses key 1 with a new initial value.
        colors.install(
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            2,
            Epoch(5),
            vec![color(1, 0.5, 1.0)],
            None,
        );
        assert!(!colors.update(
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            1,
            Epoch(6),
            &[color(1, 0.0, 1.0)]
        ));
        assert!(!colors.update(
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            2,
            Epoch(4),
            &[color(1, 0.0, 1.0)]
        ));
        assert_eq!(colors.all()[0].value.r, 0.5);
        assert_eq!(
            colors.pipelines[&(TEST_WEBVIEW_ID, TEST_PIPELINE_ID)].epoch,
            Epoch(5)
        );
        assert!(colors.update(
            TEST_WEBVIEW_ID,
            TEST_PIPELINE_ID,
            2,
            Epoch(6),
            &[color(1, 0.0, 1.0)]
        ));
        assert_eq!(colors.all()[0].value.r, 0.0);
    }
}
