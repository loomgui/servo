/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! WebRender external-image handler for GPU-rasterized 2D canvases (Loom prototype).
//!
//! The canvas thread publishes each rendered frame as the front buffer of a surfman swap chain.
//! While WebRender draws, the front surface is turned into a GL texture on WebRender's context
//! and returned to the swap chain afterwards, the same way WebGL canvases are presented.

use std::rc::Rc;

use euclid::default::Size2D;
use log::warn;
use paint_api::rendering_context::RenderingContext;
use paint_api::{ExternalImageSource, WebRenderExternalImageApi};
use rustc_hash::FxHashMap;
use surfman::chains::{SwapChainAPI, SwapChains, SwapChainsAPI};
use surfman::{Device, SurfaceTexture};
use webrender_api::ExternalImageId;

pub(crate) struct GpuCanvasExternalImages {
    rendering_context: Rc<dyn RenderingContext>,
    swap_chains: SwapChains<ExternalImageId, Device>,
    locked: FxHashMap<u64, SurfaceTexture>,
}

impl GpuCanvasExternalImages {
    pub(crate) fn new(
        rendering_context: Rc<dyn RenderingContext>,
        swap_chains: SwapChains<ExternalImageId, Device>,
    ) -> Self {
        Self {
            rendering_context,
            swap_chains,
            locked: FxHashMap::default(),
        }
    }
}

impl WebRenderExternalImageApi for GpuCanvasExternalImages {
    fn lock(&mut self, id: u64) -> (ExternalImageSource<'_>, Size2D<i32>) {
        let Some(surface) = self
            .swap_chains
            .get(ExternalImageId(id))
            .and_then(|swap_chain| swap_chain.take_surface())
        else {
            return (ExternalImageSource::NativeTexture(0), Size2D::zero());
        };
        match self.rendering_context.create_texture(surface) {
            Some((surface_texture, texture, size)) => {
                self.locked.insert(id, surface_texture);
                (ExternalImageSource::NativeTexture(texture), size)
            },
            None => {
                warn!("GPU canvas: could not create a texture for canvas image {id}");
                (ExternalImageSource::NativeTexture(0), Size2D::zero())
            },
        }
    }

    fn unlock(&mut self, id: u64) {
        let Some(surface_texture) = self.locked.remove(&id) else {
            return;
        };
        let Some(surface) = self.rendering_context.destroy_texture(surface_texture) else {
            return;
        };
        match self.swap_chains.get(ExternalImageId(id)) {
            Some(swap_chain) => swap_chain.recycle_surface(surface),
            // The canvas was destroyed while WebRender sampled it. Its device is gone, so the
            // surface cannot be destroyed here; leak it rather than panic (prototype).
            None => std::mem::forget(surface),
        }
    }
}
