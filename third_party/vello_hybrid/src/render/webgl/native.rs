// Copyright 2026 the Loom Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Native OpenGL ES 3.0 stand-in for the subset of `web_sys::WebGl2RenderingContext` that the
//! WebGL2 backend uses.
//!
//! The renderer was written against the browser API. This module mirrors the method names,
//! argument shapes and handle semantics it relies on and forwards every call to `glow`, so the
//! backend can run on a native GLES context (for example ANGLE through surfman) with only its
//! imports changed. The caller owns the context and must keep it current on the thread that uses
//! the renderer.

#![allow(
    unsafe_code,
    reason = "every call forwards to a GL function on the context the caller keeps current"
)]
#![allow(
    clippy::too_many_arguments,
    missing_docs,
    unreachable_pub,
    reason = "mirrors the browser WebGL2 API surface used by the backend"
)]

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::Cell;
use core::fmt;

use glow::HasContext;

pub type WebGlTexture = glow::NativeTexture;
pub type WebGlBuffer = glow::NativeBuffer;
pub type WebGlFramebuffer = glow::NativeFramebuffer;
pub type WebGlProgram = glow::NativeProgram;
pub type WebGlShader = glow::NativeShader;
pub type WebGlVertexArrayObject = glow::NativeVertexArray;
pub type WebGlUniformLocation = glow::NativeUniformLocation;

/// A cloneable handle to a native GLES context, shaped like the browser's WebGL2 context.
#[derive(Clone)]
pub struct WebGl2RenderingContext {
    gl: Rc<glow::Context>,
    drawing_buffer: Rc<Cell<(i32, i32)>>,
}

impl fmt::Debug for WebGl2RenderingContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WebGl2RenderingContext")
            .field("drawing_buffer", &self.drawing_buffer.get())
            .finish_non_exhaustive()
    }
}

/// Values returned by [`WebGl2RenderingContext::get_parameter`].
#[derive(Clone, Debug)]
pub enum JsValue {
    Number(f64),
    Bool(bool),
    Framebuffer(Option<WebGlFramebuffer>),
    Texture(Option<WebGlTexture>),
    Buffer(Option<WebGlBuffer>),
    Int32Array(Vec<i32>),
}

impl JsValue {
    pub const TRUE: Self = Self::Bool(true);
    pub const FALSE: Self = Self::Bool(false);

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn dyn_into<T: JsCast>(self) -> Result<T, Self> {
        T::from_js(self.clone()).ok_or(self)
    }
}

impl From<u32> for JsValue {
    fn from(value: u32) -> Self {
        Self::Number(f64::from(value))
    }
}

/// Conversion out of a [`JsValue`], standing in for `wasm_bindgen::JsCast`.
pub trait JsCast: Sized {
    fn from_js(value: JsValue) -> Option<Self>;
}

impl JsCast for WebGlFramebuffer {
    fn from_js(value: JsValue) -> Option<Self> {
        match value {
            JsValue::Framebuffer(f) => f,
            _ => None,
        }
    }
}

impl JsCast for WebGlTexture {
    fn from_js(value: JsValue) -> Option<Self> {
        match value {
            JsValue::Texture(t) => t,
            _ => None,
        }
    }
}

impl JsCast for WebGlBuffer {
    fn from_js(value: JsValue) -> Option<Self> {
        match value {
            JsValue::Buffer(b) => b,
            _ => None,
        }
    }
}

impl JsCast for js_sys::Int32Array {
    fn from_js(value: JsValue) -> Option<Self> {
        match value {
            JsValue::Int32Array(v) => Some(js_sys::Int32Array(v)),
            _ => None,
        }
    }
}

/// Typed-array stand-ins for the `js_sys` types the backend uses.
pub mod js_sys {
    use alloc::vec::Vec;

    use super::JsValue;

    #[derive(Clone, Debug, Default)]
    pub struct Array(pub(crate) Vec<u32>);

    impl Array {
        pub fn of1(value: &JsValue) -> Self {
            Self(value.as_f64().map(|v| v as u32).into_iter().collect())
        }
    }

    #[derive(Clone, Debug)]
    pub struct Int32Array(pub(crate) Vec<i32>);

    impl Int32Array {
        pub fn to_vec(&self) -> Vec<i32> {
            self.0.clone()
        }
    }

    /// Borrowed or owned `u32` data uploaded as raw bytes.
    #[derive(Clone, Debug)]
    pub struct Uint32Array(pub(crate) Vec<u8>);

    impl Uint32Array {
        pub fn from(data: &[u32]) -> Self {
            Self(bytemuck::cast_slice(data).to_vec())
        }

        /// # Safety
        /// Kept `unsafe` to match the browser API; the native version copies the data.
        pub unsafe fn view(data: &[u32]) -> Self {
            Self::from(data)
        }
    }
}

fn unpack(pixels: Option<&[u8]>) -> glow::PixelUnpackData<'_> {
    glow::PixelUnpackData::Slice(pixels)
}

impl WebGl2RenderingContext {
    /// Wrap a native GLES 3.0 context. `width`/`height` are the size of the default framebuffer
    /// the renderer draws into (see [`Self::set_drawing_buffer_size`]).
    pub fn from_glow(gl: glow::Context, width: i32, height: i32) -> Self {
        Self {
            gl: Rc::new(gl),
            drawing_buffer: Rc::new(Cell::new((width, height))),
        }
    }

    /// Update the size of the default framebuffer, for example after binding a new surface.
    pub fn set_drawing_buffer_size(&self, width: i32, height: i32) {
        self.drawing_buffer.set((width, height));
    }

    /// The underlying `glow` context.
    pub fn glow(&self) -> &glow::Context {
        &self.gl
    }

    pub fn drawing_buffer_width(&self) -> i32 {
        self.drawing_buffer.get().0
    }

    pub fn drawing_buffer_height(&self) -> i32 {
        self.drawing_buffer.get().1
    }

    pub fn get_parameter(&self, pname: u32) -> Result<JsValue, JsValue> {
        let gl = &self.gl;
        // SAFETY: the context is current on this thread (module contract).
        Ok(unsafe {
            match pname {
                glow::FRAMEBUFFER_BINDING | glow::READ_FRAMEBUFFER_BINDING => {
                    JsValue::Framebuffer(gl.get_parameter_framebuffer(pname))
                }
                glow::TEXTURE_BINDING_2D | glow::TEXTURE_BINDING_2D_ARRAY => {
                    JsValue::Texture(gl.get_parameter_texture(pname))
                }
                glow::PIXEL_PACK_BUFFER_BINDING => JsValue::Buffer(gl.get_parameter_buffer(pname)),
                glow::DEPTH_WRITEMASK => JsValue::Bool(gl.get_parameter_bool(pname)),
                glow::VIEWPORT => {
                    let mut out = [0_i32; 4];
                    gl.get_parameter_i32_slice(pname, &mut out);
                    JsValue::Int32Array(out.to_vec())
                }
                _ => JsValue::Number(f64::from(gl.get_parameter_i32(pname))),
            }
        })
    }

    pub fn is_enabled(&self, cap: u32) -> bool {
        // SAFETY: module contract.
        unsafe { self.gl.is_enabled(cap) }
    }

    pub fn enable(&self, cap: u32) {
        // SAFETY: module contract.
        unsafe { self.gl.enable(cap) }
    }

    pub fn disable(&self, cap: u32) {
        // SAFETY: module contract.
        unsafe { self.gl.disable(cap) }
    }

    pub fn viewport(&self, x: i32, y: i32, width: i32, height: i32) {
        // SAFETY: module contract.
        unsafe { self.gl.viewport(x, y, width, height) }
    }

    pub fn scissor(&self, x: i32, y: i32, width: i32, height: i32) {
        // SAFETY: module contract.
        unsafe { self.gl.scissor(x, y, width, height) }
    }

    pub fn clear(&self, mask: u32) {
        // SAFETY: module contract.
        unsafe { self.gl.clear(mask) }
    }

    pub fn clear_color(&self, r: f32, g: f32, b: f32, a: f32) {
        // SAFETY: module contract.
        unsafe { self.gl.clear_color(r, g, b, a) }
    }

    pub fn clear_depth(&self, depth: f32) {
        // SAFETY: module contract.
        unsafe { self.gl.clear_depth_f32(depth) }
    }

    pub fn depth_func(&self, func: u32) {
        // SAFETY: module contract.
        unsafe { self.gl.depth_func(func) }
    }

    pub fn depth_mask(&self, flag: bool) {
        // SAFETY: module contract.
        unsafe { self.gl.depth_mask(flag) }
    }

    pub fn blend_func(&self, sfactor: u32, dfactor: u32) {
        // SAFETY: module contract.
        unsafe { self.gl.blend_func(sfactor, dfactor) }
    }

    pub fn flush(&self) {
        // SAFETY: module contract.
        unsafe { self.gl.flush() }
    }

    pub fn active_texture(&self, unit: u32) {
        // SAFETY: module contract.
        unsafe { self.gl.active_texture(unit) }
    }

    pub fn create_texture(&self) -> Option<WebGlTexture> {
        // SAFETY: module contract.
        unsafe { self.gl.create_texture().ok() }
    }

    pub fn delete_texture(&self, texture: Option<&WebGlTexture>) {
        if let Some(texture) = texture {
            // SAFETY: module contract.
            unsafe { self.gl.delete_texture(*texture) }
        }
    }

    pub fn bind_texture(&self, target: u32, texture: Option<&WebGlTexture>) {
        // SAFETY: module contract.
        unsafe { self.gl.bind_texture(target, texture.copied()) }
    }

    pub fn tex_parameteri(&self, target: u32, pname: u32, param: i32) {
        // SAFETY: module contract.
        unsafe { self.gl.tex_parameter_i32(target, pname, param) }
    }

    pub fn tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        &self,
        target: u32,
        level: i32,
        internal_format: i32,
        width: i32,
        height: i32,
        border: i32,
        format: u32,
        ty: u32,
        pixels: Option<&[u8]>,
    ) -> Result<(), JsValue> {
        // SAFETY: module contract; GL copies client memory before returning.
        unsafe {
            self.gl.tex_image_2d(
                target,
                level,
                internal_format,
                width,
                height,
                border,
                format,
                ty,
                unpack(pixels),
            );
        }
        Ok(())
    }

    pub fn tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
        &self,
        target: u32,
        level: i32,
        internal_format: i32,
        width: i32,
        height: i32,
        border: i32,
        format: u32,
        ty: u32,
        pixels: Option<&js_sys::Uint32Array>,
    ) -> Result<(), JsValue> {
        self.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            target,
            level,
            internal_format,
            width,
            height,
            border,
            format,
            ty,
            pixels.map(|p| p.0.as_slice()),
        )
    }

    pub fn tex_image_3d_with_opt_u8_array(
        &self,
        target: u32,
        level: i32,
        internal_format: i32,
        width: i32,
        height: i32,
        depth: i32,
        border: i32,
        format: u32,
        ty: u32,
        pixels: Option<&[u8]>,
    ) -> Result<(), JsValue> {
        // SAFETY: module contract; GL copies client memory before returning.
        unsafe {
            self.gl.tex_image_3d(
                target,
                level,
                internal_format,
                width,
                height,
                depth,
                border,
                format,
                ty,
                unpack(pixels),
            );
        }
        Ok(())
    }

    pub fn tex_sub_image_3d_with_opt_u8_array(
        &self,
        target: u32,
        level: i32,
        x_offset: i32,
        y_offset: i32,
        z_offset: i32,
        width: i32,
        height: i32,
        depth: i32,
        format: u32,
        ty: u32,
        pixels: Option<&[u8]>,
    ) -> Result<(), JsValue> {
        // SAFETY: module contract; GL copies client memory before returning.
        unsafe {
            self.gl.tex_sub_image_3d(
                target,
                level,
                x_offset,
                y_offset,
                z_offset,
                width,
                height,
                depth,
                format,
                ty,
                unpack(pixels),
            );
        }
        Ok(())
    }

    pub fn copy_tex_sub_image_3d(
        &self,
        target: u32,
        level: i32,
        x_offset: i32,
        y_offset: i32,
        z_offset: i32,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
    ) {
        // SAFETY: module contract.
        unsafe {
            self.gl.copy_tex_sub_image_3d(
                target, level, x_offset, y_offset, z_offset, x, y, width, height,
            );
        }
    }

    pub fn create_buffer(&self) -> Option<WebGlBuffer> {
        // SAFETY: module contract.
        unsafe { self.gl.create_buffer().ok() }
    }

    pub fn delete_buffer(&self, buffer: Option<&WebGlBuffer>) {
        if let Some(buffer) = buffer {
            // SAFETY: module contract.
            unsafe { self.gl.delete_buffer(*buffer) }
        }
    }

    pub fn bind_buffer(&self, target: u32, buffer: Option<&WebGlBuffer>) {
        // SAFETY: module contract.
        unsafe { self.gl.bind_buffer(target, buffer.copied()) }
    }

    pub fn bind_buffer_base(&self, target: u32, index: u32, buffer: Option<&WebGlBuffer>) {
        // SAFETY: module contract.
        unsafe { self.gl.bind_buffer_base(target, index, buffer.copied()) }
    }

    pub fn buffer_data_with_i32(&self, target: u32, size: i32, usage: u32) {
        // SAFETY: module contract.
        unsafe { self.gl.buffer_data_size(target, size, usage) }
    }

    pub fn buffer_data_with_u8_array(&self, target: u32, data: &[u8], usage: u32) {
        // SAFETY: module contract; GL copies client memory before returning.
        unsafe { self.gl.buffer_data_u8_slice(target, data, usage) }
    }

    pub fn buffer_sub_data_with_i32_and_u8_array(&self, target: u32, offset: i32, data: &[u8]) {
        // SAFETY: module contract; GL copies client memory before returning.
        unsafe { self.gl.buffer_sub_data_u8_slice(target, offset, data) }
    }

    pub fn create_framebuffer(&self) -> Option<WebGlFramebuffer> {
        // SAFETY: module contract.
        unsafe { self.gl.create_framebuffer().ok() }
    }

    pub fn delete_framebuffer(&self, framebuffer: Option<&WebGlFramebuffer>) {
        if let Some(framebuffer) = framebuffer {
            // SAFETY: module contract.
            unsafe { self.gl.delete_framebuffer(*framebuffer) }
        }
    }

    pub fn bind_framebuffer(&self, target: u32, framebuffer: Option<&WebGlFramebuffer>) {
        // SAFETY: module contract.
        unsafe { self.gl.bind_framebuffer(target, framebuffer.copied()) }
    }

    pub fn framebuffer_texture_2d(
        &self,
        target: u32,
        attachment: u32,
        tex_target: u32,
        texture: Option<&WebGlTexture>,
        level: i32,
    ) {
        // SAFETY: module contract.
        unsafe {
            self.gl
                .framebuffer_texture_2d(target, attachment, tex_target, texture.copied(), level);
        }
    }

    pub fn framebuffer_texture_layer(
        &self,
        target: u32,
        attachment: u32,
        texture: Option<&WebGlTexture>,
        level: i32,
        layer: i32,
    ) {
        // SAFETY: module contract.
        unsafe {
            self.gl
                .framebuffer_texture_layer(target, attachment, texture.copied(), level, layer);
        }
    }

    pub fn invalidate_framebuffer(
        &self,
        target: u32,
        attachments: &js_sys::Array,
    ) -> Result<(), JsValue> {
        // SAFETY: module contract.
        unsafe { self.gl.invalidate_framebuffer(target, &attachments.0) }
        Ok(())
    }

    pub fn create_vertex_array(&self) -> Option<WebGlVertexArrayObject> {
        // SAFETY: module contract.
        unsafe { self.gl.create_vertex_array().ok() }
    }

    pub fn delete_vertex_array(&self, vao: Option<&WebGlVertexArrayObject>) {
        if let Some(vao) = vao {
            // SAFETY: module contract.
            unsafe { self.gl.delete_vertex_array(*vao) }
        }
    }

    pub fn bind_vertex_array(&self, vao: Option<&WebGlVertexArrayObject>) {
        // SAFETY: module contract.
        unsafe { self.gl.bind_vertex_array(vao.copied()) }
    }

    pub fn enable_vertex_attrib_array(&self, index: u32) {
        // SAFETY: module contract.
        unsafe { self.gl.enable_vertex_attrib_array(index) }
    }

    pub fn vertex_attrib_divisor(&self, index: u32, divisor: u32) {
        // SAFETY: module contract.
        unsafe { self.gl.vertex_attrib_divisor(index, divisor) }
    }

    pub fn vertex_attrib_i_pointer_with_i32(
        &self,
        index: u32,
        size: i32,
        ty: u32,
        stride: i32,
        offset: i32,
    ) {
        // SAFETY: module contract.
        unsafe {
            self.gl
                .vertex_attrib_pointer_i32(index, size, ty, stride, offset);
        }
    }

    pub fn draw_arrays_instanced(&self, mode: u32, first: i32, count: i32, instances: i32) {
        // SAFETY: module contract.
        unsafe { self.gl.draw_arrays_instanced(mode, first, count, instances) }
    }

    pub fn create_shader(&self, ty: u32) -> Option<WebGlShader> {
        // SAFETY: module contract.
        unsafe { self.gl.create_shader(ty).ok() }
    }

    pub fn delete_shader(&self, shader: Option<&WebGlShader>) {
        if let Some(shader) = shader {
            // SAFETY: module contract.
            unsafe { self.gl.delete_shader(*shader) }
        }
    }

    pub fn shader_source(&self, shader: &WebGlShader, source: &str) {
        // SAFETY: module contract.
        unsafe { self.gl.shader_source(*shader, source) }
    }

    pub fn compile_shader(&self, shader: &WebGlShader) {
        // SAFETY: module contract.
        unsafe { self.gl.compile_shader(*shader) }
    }

    pub fn get_shader_parameter(&self, shader: &WebGlShader, pname: u32) -> JsValue {
        // SAFETY: module contract.
        unsafe {
            match pname {
                glow::COMPILE_STATUS => JsValue::Bool(self.gl.get_shader_compile_status(*shader)),
                _ => JsValue::Number(0.0),
            }
        }
    }

    pub fn get_shader_info_log(&self, shader: &WebGlShader) -> Option<String> {
        // SAFETY: module contract.
        Some(unsafe { self.gl.get_shader_info_log(*shader) })
    }

    pub fn create_program(&self) -> Option<WebGlProgram> {
        // SAFETY: module contract.
        unsafe { self.gl.create_program().ok() }
    }

    pub fn delete_program(&self, program: Option<&WebGlProgram>) {
        if let Some(program) = program {
            // SAFETY: module contract.
            unsafe { self.gl.delete_program(*program) }
        }
    }

    pub fn attach_shader(&self, program: &WebGlProgram, shader: &WebGlShader) {
        // SAFETY: module contract.
        unsafe { self.gl.attach_shader(*program, *shader) }
    }

    pub fn link_program(&self, program: &WebGlProgram) {
        // SAFETY: module contract.
        unsafe { self.gl.link_program(*program) }
    }

    pub fn get_program_parameter(&self, program: &WebGlProgram, pname: u32) -> JsValue {
        // SAFETY: module contract.
        unsafe {
            match pname {
                glow::LINK_STATUS => JsValue::Bool(self.gl.get_program_link_status(*program)),
                _ => JsValue::Number(0.0),
            }
        }
    }

    pub fn get_program_info_log(&self, program: &WebGlProgram) -> Option<String> {
        // SAFETY: module contract.
        Some(unsafe { self.gl.get_program_info_log(*program) })
    }

    pub fn use_program(&self, program: Option<&WebGlProgram>) {
        // SAFETY: module contract.
        unsafe { self.gl.use_program(program.copied()) }
    }

    pub fn get_uniform_location(
        &self,
        program: &WebGlProgram,
        name: &str,
    ) -> Option<WebGlUniformLocation> {
        // SAFETY: module contract.
        unsafe { self.gl.get_uniform_location(*program, name) }
    }

    pub fn uniform1i(&self, location: Option<&WebGlUniformLocation>, value: i32) {
        // SAFETY: module contract.
        unsafe { self.gl.uniform_1_i32(location, value) }
    }

    pub fn get_uniform_block_index(&self, program: &WebGlProgram, name: &str) -> u32 {
        // SAFETY: module contract.
        unsafe { self.gl.get_uniform_block_index(*program, name) }.unwrap_or(glow::INVALID_INDEX)
    }

    pub fn uniform_block_binding(&self, program: &WebGlProgram, index: u32, binding: u32) {
        // SAFETY: module contract.
        unsafe { self.gl.uniform_block_binding(*program, index, binding) }
    }
}

/// `WebGl2RenderingContext::NAME` constants used by the backend, mapped to GL enums.
#[allow(non_upper_case_globals, reason = "mirrors the browser constant names")]
impl WebGl2RenderingContext {
    pub const ACTIVE_TEXTURE: u32 = glow::ACTIVE_TEXTURE;
    pub const ARRAY_BUFFER: u32 = glow::ARRAY_BUFFER;
    pub const BLEND: u32 = glow::BLEND;
    pub const CLAMP_TO_EDGE: u32 = glow::CLAMP_TO_EDGE;
    pub const COLOR_ATTACHMENT0: u32 = glow::COLOR_ATTACHMENT0;
    pub const COLOR_BUFFER_BIT: u32 = glow::COLOR_BUFFER_BIT;
    pub const COMPILE_STATUS: u32 = glow::COMPILE_STATUS;
    pub const DEPTH: u32 = glow::DEPTH;
    pub const DEPTH_BITS: u32 = glow::DEPTH_BITS;
    pub const DEPTH_BUFFER_BIT: u32 = glow::DEPTH_BUFFER_BIT;
    pub const DEPTH_TEST: u32 = glow::DEPTH_TEST;
    pub const DEPTH_WRITEMASK: u32 = glow::DEPTH_WRITEMASK;
    pub const DYNAMIC_DRAW: u32 = glow::DYNAMIC_DRAW;
    pub const FRAGMENT_SHADER: u32 = glow::FRAGMENT_SHADER;
    pub const FRAMEBUFFER: u32 = glow::FRAMEBUFFER;
    pub const FRAMEBUFFER_BINDING: u32 = glow::FRAMEBUFFER_BINDING;
    pub const INVALID_INDEX: u32 = glow::INVALID_INDEX;
    pub const LEQUAL: u32 = glow::LEQUAL;
    pub const LINEAR: u32 = glow::LINEAR;
    pub const LINK_STATUS: u32 = glow::LINK_STATUS;
    pub const MAX_ARRAY_TEXTURE_LAYERS: u32 = glow::MAX_ARRAY_TEXTURE_LAYERS;
    pub const MAX_TEXTURE_SIZE: u32 = glow::MAX_TEXTURE_SIZE;
    pub const NEAREST: u32 = glow::NEAREST;
    pub const ONE: u32 = glow::ONE;
    pub const ONE_MINUS_SRC_ALPHA: u32 = glow::ONE_MINUS_SRC_ALPHA;
    pub const PIXEL_PACK_BUFFER: u32 = glow::PIXEL_PACK_BUFFER;
    pub const PIXEL_PACK_BUFFER_BINDING: u32 = glow::PIXEL_PACK_BUFFER_BINDING;
    pub const READ_FRAMEBUFFER: u32 = glow::READ_FRAMEBUFFER;
    pub const READ_FRAMEBUFFER_BINDING: u32 = glow::READ_FRAMEBUFFER_BINDING;
    pub const RGBA: u32 = glow::RGBA;
    pub const RGBA32UI: u32 = glow::RGBA32UI;
    pub const RGBA8: u32 = glow::RGBA8;
    pub const RGBA_INTEGER: u32 = glow::RGBA_INTEGER;
    pub const SCISSOR_TEST: u32 = glow::SCISSOR_TEST;
    pub const STATIC_DRAW: u32 = glow::STATIC_DRAW;
    pub const TEXTURE0: u32 = glow::TEXTURE0;
    pub const TEXTURE1: u32 = glow::TEXTURE1;
    pub const TEXTURE2: u32 = glow::TEXTURE2;
    pub const TEXTURE3: u32 = glow::TEXTURE3;
    pub const TEXTURE4: u32 = glow::TEXTURE4;
    pub const TEXTURE_2D: u32 = glow::TEXTURE_2D;
    pub const TEXTURE_2D_ARRAY: u32 = glow::TEXTURE_2D_ARRAY;
    pub const TEXTURE_BINDING_2D: u32 = glow::TEXTURE_BINDING_2D;
    pub const TEXTURE_BINDING_2D_ARRAY: u32 = glow::TEXTURE_BINDING_2D_ARRAY;
    pub const TEXTURE_MAG_FILTER: u32 = glow::TEXTURE_MAG_FILTER;
    pub const TEXTURE_MAX_LEVEL: u32 = glow::TEXTURE_MAX_LEVEL;
    pub const TEXTURE_MIN_FILTER: u32 = glow::TEXTURE_MIN_FILTER;
    pub const TEXTURE_WRAP_S: u32 = glow::TEXTURE_WRAP_S;
    pub const TEXTURE_WRAP_T: u32 = glow::TEXTURE_WRAP_T;
    pub const TRIANGLE_STRIP: u32 = glow::TRIANGLE_STRIP;
    pub const UNIFORM_BUFFER: u32 = glow::UNIFORM_BUFFER;
    pub const UNSIGNED_BYTE: u32 = glow::UNSIGNED_BYTE;
    pub const UNSIGNED_INT: u32 = glow::UNSIGNED_INT;
    pub const VERTEX_SHADER: u32 = glow::VERTEX_SHADER;
    pub const VIEWPORT: u32 = glow::VIEWPORT;
}
