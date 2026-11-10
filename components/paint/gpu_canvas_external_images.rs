/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! WebRender external-image handler for GPU-rasterized 2D canvases (Loom prototype).
//!
//! The canvas thread publishes each rendered frame as the front buffer of a surfman swap chain.
//! When the painter prepares a frame it latches every canvas's front buffer, so the frame shows
//! the canvas content that belongs to it even if the canvas presents again before the frame is
//! drawn. While WebRender draws, a latched surface is turned into a GL texture on WebRender's
//! context, and it returns to the swap chain afterwards, the same way WebGL canvases are
//! presented.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use euclid::default::Size2D;
use log::warn;
use paint_api::rendering_context::RenderingContext;
use paint_api::{ExternalImageSource, WebRenderExternalImageApi};
use rustc_hash::FxHashMap;
use surfman::chains::{SwapChainAPI, SwapChains, SwapChainsAPI};
use surfman::{Device, Surface, SurfaceTexture};
use webrender_api::ExternalImageId;

/// The canvas front buffers held for the frame that is prepared but not drawn yet.
///
/// Shared by the painter, which latches when it prepares a frame and releases once the frame was
/// drawn or discarded, and the external-image handler, which hands the latched surfaces to
/// WebRender.
pub(crate) struct GpuCanvasLatch {
    swap_chains: SwapChains<ExternalImageId, Device>,
    canvas_image_ids: Arc<Mutex<Vec<ExternalImageId>>>,
    latched: RefCell<FxHashMap<u64, Surface>>,
}

impl GpuCanvasLatch {
    pub(crate) fn new(
        swap_chains: SwapChains<ExternalImageId, Device>,
        canvas_image_ids: Arc<Mutex<Vec<ExternalImageId>>>,
    ) -> Self {
        Self {
            swap_chains,
            canvas_image_ids,
            latched: RefCell::default(),
        }
    }

    /// Take the current front buffer of every GPU canvas for the frame being prepared.
    pub(crate) fn latch(&self) {
        self.release();
        let ids = self
            .canvas_image_ids
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        let mut latched = self.latched.borrow_mut();
        for id in ids {
            let surface = self
                .swap_chains
                .get(id)
                .and_then(|swap_chain| swap_chain.take_surface());
            if let Some(surface) = surface {
                latched.insert(id.0, surface);
            }
        }
    }

    /// Return every surface that is still latched to its swap chain.
    pub(crate) fn release(&self) {
        let surfaces: Vec<_> = self.latched.borrow_mut().drain().collect();
        for (id, surface) in surfaces {
            self.give_back(id, surface);
        }
    }

    /// The surface to draw for `id`: the latched one, or the canvas's current front buffer when
    /// the canvas was not latched (it was created after the frame was prepared).
    fn take(&self, id: u64) -> Option<Surface> {
        if let Some(surface) = self.latched.borrow_mut().remove(&id) {
            return Some(surface);
        }
        self.swap_chains
            .get(ExternalImageId(id))
            .and_then(|swap_chain| swap_chain.take_surface())
    }

    fn give_back(&self, id: u64, surface: Surface) {
        match self.swap_chains.get(ExternalImageId(id)) {
            Some(swap_chain) => swap_chain.recycle_surface(surface),
            // The canvas was destroyed while the frame held its surface. Its device is gone, so
            // the surface cannot be destroyed here; leak it rather than panic (prototype).
            None => std::mem::forget(surface),
        }
    }
}

pub(crate) struct GpuCanvasExternalImages {
    rendering_context: Rc<dyn RenderingContext>,
    latch: Rc<GpuCanvasLatch>,
    locked: FxHashMap<u64, SurfaceTexture>,
}

impl GpuCanvasExternalImages {
    pub(crate) fn new(
        rendering_context: Rc<dyn RenderingContext>,
        latch: Rc<GpuCanvasLatch>,
    ) -> Self {
        Self {
            rendering_context,
            latch,
            locked: FxHashMap::default(),
        }
    }
}

impl WebRenderExternalImageApi for GpuCanvasExternalImages {
    fn lock(&mut self, id: u64) -> (ExternalImageSource<'_>, Size2D<i32>) {
        let Some(surface) = self.latch.take(id) else {
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
        self.latch.give_back(id, surface);
    }
}
