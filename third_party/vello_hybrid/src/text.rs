// Copyright 2026 the Vello Authors and the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello Hybrid (GPU) glyph rendering backend.
//!
//! Provides [`GlyphAtlas`] and the [`GlyphRenderer`](glifo::GlyphRenderer)
//! implementation for [`Scene`].
//!
//! Unlike the CPU backend, no local `Pixmap` storage is allocated here — the
//! GPU renderer owns atlas textures and receives pixel data through the
//! pending-upload queue.

use core::ops::RangeInclusive;

use crate::{AtlasId, PathStripGenerator, PathStripSettings, PathStrips, Resources, Scene};
use alloc::vec::Vec;
use glifo::GlyphPrepCache;
use glifo::atlas::{PendingBitmapUpload, PendingClearRect};
use glifo::renderer::replay_atlas_commands;
use glifo::{
    AtlasCacher, AtlasSlot, DrawSink, GLYPH_PADDING, Glyph, GlyphAtlas, GlyphCacheConfig,
    GlyphRunBackend, ImageCache,
};
use peniko::BlendMode;
use peniko::color::palette::css::BLACK;
use peniko::color::{AlphaColor, Srgb};
use vello_common::kurbo::{Affine, BezPath, Rect, Stroke};
use vello_common::multi_atlas::AtlasConfig;
use vello_common::paint::{Image, ImageSource, PaintType};
use vello_common::peniko;
use vello_common::peniko::{Fill, FontData};
use vello_common::transforms::Transforms;

/// Glyph atlas cache for the hybrid (GPU) renderer.
#[derive(Debug)]
pub(crate) struct GlyphAtlasResources {
    pub(crate) glyph_atlas: GlyphAtlas,
    pub(crate) glyph_renderer: Scene,
}

impl GlyphAtlasResources {
    pub(crate) fn with_config(
        atlas_width: u16,
        atlas_height: u16,
        eviction_config: GlyphCacheConfig,
    ) -> Self {
        Self {
            glyph_atlas: GlyphAtlas::with_config(eviction_config),
            glyph_renderer: Scene::new(atlas_width, atlas_height),
        }
    }

    pub(crate) fn maintain(&mut self, image_cache: &mut ImageCache) {
        self.glyph_atlas.maintain(image_cache);
    }
}

impl Resources {
    fn ensure_glyph_resources(&mut self) {
        if self.glyph_resources.is_none() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "atlas dimensions are configured to fit in u16"
            )]
            let (atlas_width, atlas_height) = {
                let (width, height) = self.image_cache.atlas_manager().config().atlas_size;
                (width as u16, height as u16)
            };
            // TODO: Use a config optionally provided by the user!
            self.glyph_resources = Some(GlyphAtlasResources::with_config(
                atlas_width,
                atlas_height,
                GlyphCacheConfig::default(),
            ));
        }
    }

    fn atlas_config(&self) -> AtlasConfig {
        *self.image_cache.atlas_manager().config()
    }

    fn atlas_count(&self) -> u32 {
        u32::try_from(self.image_cache.atlas_count()).unwrap()
    }

    fn replay_pending_atlas_commands(&mut self, mut f: impl FnMut(&Scene, AtlasId)) {
        if let Some(glyph_resources) = self.glyph_resources.as_mut() {
            glyph_resources
                .glyph_atlas
                .replay_pending_atlas_commands(|recorder| {
                    glyph_resources.glyph_renderer.reset();
                    replay_atlas_commands(
                        &mut recorder.commands,
                        &mut glyph_resources.glyph_renderer,
                    );
                    f(
                        &glyph_resources.glyph_renderer,
                        AtlasId::new(recorder.page_index),
                    );
                });
        }
    }

    pub(crate) fn before_render<T>(
        &mut self,
        backend: &mut T,
        mut render_to_atlas: impl FnMut(&mut T, &Scene, u32, AtlasConfig, AtlasId),
        mut upload_to_atlas: impl FnMut(&mut T, &ImageCache, &PendingBitmapUpload, u32, u32),
    ) {
        let atlas_count = self.atlas_count();
        let atlas_config = self.atlas_config();
        self.replay_pending_atlas_commands(|glyph_renderer, atlas_id| {
            render_to_atlas(backend, glyph_renderer, atlas_count, atlas_config, atlas_id);
        });

        const PADDING: u32 = GLYPH_PADDING as u32;

        if let Some(glyph_resources) = self.glyph_resources.as_mut() {
            for upload in glyph_resources.glyph_atlas.drain_pending_uploads() {
                let resource = self.image_cache.get(upload.image_id).unwrap();
                let dst_x = resource.offset[0] as u32 + PADDING;
                let dst_y = resource.offset[1] as u32 + PADDING;
                upload_to_atlas(backend, &self.image_cache, &upload, dst_x, dst_y);
            }
        }
    }

    pub(crate) fn after_render<T>(
        &mut self,
        backend: &mut T,
        mut clear_rect: impl FnMut(&mut T, &PendingClearRect),
    ) {
        self.glyph_prep_cache.maintain();
        if let Some(glyph_resources) = self.glyph_resources.as_mut() {
            glyph_resources.maintain(&mut self.image_cache);
            for rect in glyph_resources.glyph_atlas.drain_pending_clear_rects() {
                clear_rect(backend, &rect);
            }
        }
    }
}

/// [`DrawSink`] for the hybrid [`Scene`].
impl DrawSink for Scene {
    #[inline]
    fn set_transform(&mut self, t: Affine) {
        Self::set_transform(self, t);
    }

    #[inline]
    fn set_paint(&mut self, paint: glifo::AtlasPaint) {
        Self::set_paint(self, paint);
    }

    #[inline]
    fn set_paint_transform(&mut self, t: Affine) {
        Self::set_paint_transform(self, t);
    }

    #[inline]
    fn fill_path(&mut self, path: &BezPath) {
        Self::fill_path(self, path);
    }

    #[inline]
    fn fill_rect(&mut self, rect: &Rect) {
        Self::fill_rect(self, rect);
    }

    #[inline]
    fn push_clip_layer(&mut self, clip: &BezPath) {
        Self::push_clip_layer(self, clip);
    }

    #[inline]
    fn push_clip_path(&mut self, clip: &BezPath) {
        Self::push_clip_path(self, clip);
    }

    #[inline]
    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        Self::push_blend_layer(self, blend_mode);
    }

    #[inline]
    fn pop_layer(&mut self) {
        Self::pop_layer(self);
    }

    #[inline]
    fn pop_clip_path(&mut self) {
        Self::pop_clip_path(self);
    }

    #[inline]
    fn width(&self) -> u16 {
        Self::width(self)
    }

    #[inline]
    fn height(&self) -> u16 {
        Self::height(self)
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct HybridGlyphRunBackend<'a> {
    pub(crate) scene: &'a mut Scene,
    pub(crate) resources: &'a mut Resources,
    pub(crate) atlas_cache_enabled: bool,
}

impl<'a> HybridGlyphRunBackend<'a> {
    fn render_glyphs<Glyphs>(
        self,
        run: glifo::GlyphRun<'a>,
        glyphs: Glyphs,
        render: impl FnOnce(&mut glifo::GlyphRunRenderer<'a, 'a, Glyphs>, &mut Scene),
    ) where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let atlas_cacher = if self.atlas_cache_enabled {
            self.resources.ensure_glyph_resources();
            let glyph_resources = self
                .resources
                .glyph_resources
                .as_mut()
                .expect("glyph atlas resources must exist after initialization");
            AtlasCacher::Enabled(
                &mut glyph_resources.glyph_atlas,
                &mut self.resources.image_cache,
            )
        } else {
            AtlasCacher::Disabled
        };

        let mut glyph_run = run.build(
            glyphs,
            self.resources.glyph_prep_cache.as_mut(),
            atlas_cacher,
        );
        render(&mut glyph_run, self.scene);
    }
}

impl<'a> GlyphRunBackend<'a> for HybridGlyphRunBackend<'a> {
    fn atlas_cache(mut self, enabled: bool) -> Self {
        self.atlas_cache_enabled = enabled;
        self
    }

    fn fill_glyphs<Glyphs>(self, run: glifo::GlyphRun<'a>, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        self.render_glyphs(run, glyphs, |glyph_run, scene| glyph_run.fill_glyphs(scene));
    }

    fn stroke_glyphs<Glyphs>(self, run: glifo::GlyphRun<'a>, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        self.render_glyphs(run, glyphs, |glyph_run, scene| {
            let stroke_adjustment = glyph_run.stroke_adjustment();
            let original_width = scene.stroke().width;
            scene.stroke_mut().width *= stroke_adjustment;
            glyph_run.stroke_glyphs(scene);
            scene.stroke_mut().width = original_width;
        });
    }

    fn render_decoration<Glyphs>(
        self,
        run: glifo::GlyphRun<'a>,
        glyphs: Glyphs,
        x_range: RangeInclusive<f32>,
        baseline_y: f32,
        offset: f32,
        size: f32,
        buffer: f32,
    ) where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        self.render_glyphs(run, glyphs, |glyph_run, scene| {
            glyph_run.render_decoration(x_range, baseline_y, offset, size, buffer, scene);
        });
    }
}

/// A glyph run builder.
pub type GlyphRunBuilder<'a> = glifo::GlyphRunBuilder<'a, HybridGlyphRunBackend<'a>>;

impl glifo::GlyphRenderer for Scene {
    type SavedState = vello_common::render_state::RenderState;

    #[inline]
    fn save_state(&mut self) -> Self::SavedState {
        self.save_current_state()
    }

    #[inline]
    fn restore_state(&mut self, state: Self::SavedState) {
        Self::restore_state(self, state);
    }

    #[inline]
    fn stroke_path(&mut self, path: &BezPath) {
        Self::stroke_path(self, path);
    }

    #[inline]
    fn set_paint_image(&mut self, image: Image) {
        self.set_paint(image);
    }

    #[inline]
    fn set_tint(&mut self, tint: Option<vello_common::paint::Tint>) {
        Self::set_tint(self, tint);
    }

    #[inline]
    fn get_context_color(&self) -> AlphaColor<Srgb> {
        let paint = self.paint().clone();
        match paint {
            PaintType::Solid(s) => s,
            _ => BLACK,
        }
    }

    #[inline]
    fn current_paint(&self) -> &PaintType {
        self.paint()
    }

    #[inline]
    fn atlas_image_source(&self, atlas_slot: &AtlasSlot) -> ImageSource {
        ImageSource::opaque_id(atlas_slot.image_id)
    }

    #[inline]
    fn atlas_paint_transform(&self, _atlas_slot: &AtlasSlot) -> Affine {
        let padding = GLYPH_PADDING as f64;
        Affine::translate((-padding, -padding))
    }
}

/// Loom: a glyph run whose outlines are turned into strips away from the scene, on any thread;
/// see [`Scene::glyph_run_job`]. Preparing the glyphs and generating their strips is nearly all
/// the work of drawing text.
#[derive(Clone, Debug)]
pub struct GlyphRunJob {
    font: FontData,
    font_size: f32,
    glyphs: Vec<Glyph>,
    /// The stroke of stroked text, as set on the scene; filled text has none.
    stroke: Option<Stroke>,
    fill_rule: Fill,
    transform: Affine,
    paint_transform: Affine,
    root_transform: Affine,
    width: u16,
    height: u16,
    level: vello_common::fearless_simd::Level,
    aliasing_threshold: Option<u8>,
}

/// Loom: what [`GlyphRunJob::strips`] made of a glyph run.
#[derive(Debug)]
pub struct GlyphRunStrips(GlyphRunResult);

#[derive(Debug)]
enum GlyphRunResult {
    /// The transforms and strips of each glyph outline, in order.
    Outlines(Vec<(Transforms, PathStrips)>),
    /// The run has glyphs that are not plain outlines (bitmaps, color layers); it is drawn
    /// into the scene as usual.
    Scene(GlyphRunJob),
}

/// Loom: what running glyph run jobs keeps between them: glyph preparation caches and a strip
/// generator. Use one per thread, for as long as possible.
#[derive(Debug, Default)]
pub struct GlyphRunCaches {
    prep: GlyphPrepCache,
    generator: PathStripGenerator,
    jobs: u32,
}

/// Glyph run jobs a thread runs between two maintenances of its glyph caches.
const JOBS_PER_CACHE_MAINTENANCE: u32 = 32;

impl GlyphRunJob {
    /// The strips of every glyph outline of the run, as [`Scene::glyph_run`] would generate
    /// them. Runs on any thread, with that thread's glyph caches.
    pub fn strips(&self, caches: &mut GlyphRunCaches) -> GlyphRunStrips {
        let GlyphRunCaches {
            prep,
            generator,
            jobs,
        } = caches;
        let mut collector = OutlineCollector {
            job: self,
            transforms: Transforms::new(),
            stroke: self.stroke.clone(),
            generator,
            outlines: Vec::new(),
            other: false,
            paint: PaintType::Solid(BLACK),
        };
        collector.transforms.set_transform(self.transform);
        collector
            .transforms
            .set_paint_transform(self.paint_transform);
        let builder = glifo::GlyphRunBuilder::new(
            self.font.clone(),
            self.transform,
            self.paint_transform,
            OutlineBackend {
                cache: prep,
                collector: &mut collector,
            },
        )
        .font_size(self.font_size);
        if self.stroke.is_some() {
            builder.stroke_glyphs(self.glyphs.iter().copied());
        } else {
            builder.fill_glyphs(self.glyphs.iter().copied());
        }
        *jobs += 1;
        if *jobs % JOBS_PER_CACHE_MAINTENANCE == 0 {
            prep.maintain();
        }
        GlyphRunStrips(if collector.other {
            GlyphRunResult::Scene(self.clone())
        } else {
            GlyphRunResult::Outlines(collector.outlines)
        })
    }
}

impl Scene {
    /// Loom: a job that generates the strips of a glyph run with the current state, or
    /// `None` when they have to be generated in the scene: a clip path, filter effect or
    /// non-isolated blend mode is set. The scene draws the strips with
    /// [`Self::draw_glyph_run_strips`].
    pub fn glyph_run_job(
        &self,
        font: &FontData,
        font_size: f32,
        glyphs: impl Iterator<Item = Glyph>,
        stroked: bool,
    ) -> Option<GlyphRunJob> {
        if !self.can_defer_paths() {
            return None;
        }
        let settings = self.path_strip_settings()?;
        let state = self.render_state();
        Some(GlyphRunJob {
            font: font.clone(),
            font_size,
            glyphs: glyphs.collect(),
            stroke: stroked.then(|| state.stroke.clone()),
            fill_rule: state.fill_rule,
            transform: *state.transforms.transform(),
            paint_transform: *state.transforms.paint_transform(),
            root_transform: self.root_transform(),
            width: settings.width,
            height: settings.height,
            level: settings.level,
            aliasing_threshold: settings.aliasing_threshold,
        })
    }

    /// Loom: draw a glyph run from its [`GlyphRunJob::strips`] with the current paint, as
    /// [`Self::glyph_run`] would have drawn it when the job was made. The scene state must be
    /// what it was then, except for the transforms, which the job holds.
    pub fn draw_glyph_run_strips(&mut self, resources: &mut Resources, strips: &GlyphRunStrips) {
        let saved = self.render_state().transforms.clone();
        match &strips.0 {
            GlyphRunResult::Outlines(outlines) => {
                for (transforms, strips) in outlines {
                    self.render_state_mut().transforms = transforms.clone();
                    self.draw_path_strips(strips);
                }
            },
            GlyphRunResult::Scene(job) => {
                self.set_transform(job.transform);
                self.set_paint_transform(job.paint_transform);
                if let Some(stroke) = &job.stroke {
                    self.set_stroke(stroke.clone());
                }
                let run = self
                    .glyph_run(resources, &job.font)
                    .font_size(job.font_size);
                if job.stroke.is_some() {
                    run.stroke_glyphs(job.glyphs.iter().copied());
                } else {
                    run.fill_glyphs(job.glyphs.iter().copied());
                }
            },
        }
        self.render_state_mut().transforms = saved;
    }
}

/// The glifo backend of a glyph run job.
struct OutlineBackend<'a, 'j> {
    cache: &'a mut GlyphPrepCache,
    collector: &'a mut OutlineCollector<'j>,
}

impl<'a> GlyphRunBackend<'a> for OutlineBackend<'a, '_> {
    fn atlas_cache(self, _enabled: bool) -> Self {
        self
    }

    fn fill_glyphs<Glyphs>(self, run: glifo::GlyphRun<'a>, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let mut glyph_run = run.build(glyphs, self.cache.as_mut(), AtlasCacher::Disabled);
        glyph_run.fill_glyphs(self.collector);
    }

    fn stroke_glyphs<Glyphs>(self, run: glifo::GlyphRun<'a>, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let mut glyph_run = run.build(glyphs, self.cache.as_mut(), AtlasCacher::Disabled);
        // As `HybridGlyphRunBackend` does.
        if let Some(stroke) = &mut self.collector.stroke {
            stroke.width *= glyph_run.stroke_adjustment();
        }
        glyph_run.stroke_glyphs(self.collector);
    }

    fn render_decoration<Glyphs>(
        self,
        _run: glifo::GlyphRun<'a>,
        _glyphs: Glyphs,
        _x_range: RangeInclusive<f32>,
        _baseline_y: f32,
        _offset: f32,
        _size: f32,
        _buffer: f32,
    ) where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        self.collector.other = true;
    }
}

/// Collects the strips of glyph outlines. Anything else a glyph draws marks the run for the
/// scene.
struct OutlineCollector<'j> {
    job: &'j GlyphRunJob,
    transforms: Transforms,
    stroke: Option<Stroke>,
    generator: &'j mut PathStripGenerator,
    outlines: Vec<(Transforms, PathStrips)>,
    other: bool,
    paint: PaintType,
}

impl OutlineCollector<'_> {
    fn settings(&self) -> PathStripSettings {
        PathStripSettings {
            width: self.job.width,
            height: self.job.height,
            level: self.job.level,
            transform: self.job.root_transform * *self.transforms.transform(),
            aliasing_threshold: self.job.aliasing_threshold,
        }
    }
}

impl DrawSink for OutlineCollector<'_> {
    fn set_transform(&mut self, t: Affine) {
        self.transforms.set_transform(t);
    }

    fn set_paint(&mut self, _paint: glifo::AtlasPaint) {
        self.other = true;
    }

    fn set_paint_transform(&mut self, t: Affine) {
        self.transforms.set_paint_transform(t);
    }

    fn fill_path(&mut self, path: &BezPath) {
        if self.other {
            return;
        }
        let settings = self.settings();
        let strips = self.generator.fill(path, self.job.fill_rule, &settings);
        self.outlines.push((self.transforms.clone(), strips));
    }

    fn fill_rect(&mut self, _rect: &Rect) {
        self.other = true;
    }

    fn push_clip_layer(&mut self, _clip: &BezPath) {
        self.other = true;
    }

    fn push_clip_path(&mut self, _clip: &BezPath) {
        self.other = true;
    }

    fn push_blend_layer(&mut self, _blend_mode: BlendMode) {
        self.other = true;
    }

    fn pop_layer(&mut self) {
        self.other = true;
    }

    fn pop_clip_path(&mut self) {
        self.other = true;
    }

    fn width(&self) -> u16 {
        self.job.width
    }

    fn height(&self) -> u16 {
        self.job.height
    }
}

impl glifo::GlyphRenderer for OutlineCollector<'_> {
    type SavedState = Transforms;

    fn save_state(&mut self) -> Self::SavedState {
        self.transforms.clone()
    }

    fn restore_state(&mut self, state: Self::SavedState) {
        self.transforms = state;
    }

    fn stroke_path(&mut self, path: &BezPath) {
        if self.other {
            return;
        }
        let Some(stroke) = &self.stroke else {
            self.other = true;
            return;
        };
        let settings = self.settings();
        let strips = self.generator.stroke(path, stroke, &settings);
        self.outlines.push((self.transforms.clone(), strips));
    }

    fn set_paint_image(&mut self, _image: Image) {
        self.other = true;
    }

    fn set_tint(&mut self, _tint: Option<vello_common::paint::Tint>) {
        self.other = true;
    }

    fn get_context_color(&self) -> AlphaColor<Srgb> {
        BLACK
    }

    fn current_paint(&self) -> &PaintType {
        &self.paint
    }

    fn atlas_image_source(&self, atlas_slot: &AtlasSlot) -> ImageSource {
        ImageSource::opaque_id(atlas_slot.image_id)
    }

    fn atlas_paint_transform(&self, _atlas_slot: &AtlasSlot) -> Affine {
        let padding = GLYPH_PADDING as f64;
        Affine::translate((-padding, -padding))
    }
}
