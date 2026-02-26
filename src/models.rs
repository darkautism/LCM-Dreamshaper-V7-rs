use anyhow::{Context, Result};
use rknn_rs::prelude::*;
use rknn_sys_rs as sys;
use std::path::Path;
use std::ptr::null_mut;

/// Thin wrapper around an RKNN context.
pub struct RknnModel {
    pub rknn: Rknn,
}

impl RknnModel {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let rknn = Rknn::rknn_init(path_ref)
            .with_context(|| format!("Failed to load RKNN model: {}", path_ref.display()))?;
        Ok(Self { rknn })
    }

    pub fn print_info(&self) -> Result<()> {
        self.rknn.info().map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(())
    }

    /// Run inference: set inputs (all at once) then retrieve first output as f32.
    pub fn run_f32(&self, inputs: &[(usize, &[f32])]) -> Result<Vec<f32>> {
        for &(idx, data) in inputs {
            self.rknn
                .input_set_slice(idx, data, false, RknnTensorType::Float32, RknnTensorFormat::NHWC)
                .map_err(|e| anyhow::anyhow!("input_set_slice[{}] failed: {}", idx, e))?;
        }
        self.rknn.run().map_err(|e| anyhow::anyhow!("rknn_run failed: {}", e))?;
        let out = self.rknn.outputs_get::<f32>().map_err(|e| anyhow::anyhow!("outputs_get failed: {}", e))?;
        Ok(out.to_vec())
    }

    /// Run with int32 inputs.
    pub fn run_with_int32_inputs(
        &self,
        int_inputs: &[(usize, &[i32])],
        f32_inputs: &[(usize, &[f32])],
    ) -> Result<Vec<f32>> {
        for &(idx, data) in int_inputs {
            self.rknn
                .input_set_slice(idx, data, false, RknnTensorType::Int32, RknnTensorFormat::Undefined)
                .map_err(|e| anyhow::anyhow!("input_set_slice int32[{}] failed: {}", idx, e))?;
        }
        for &(idx, data) in f32_inputs {
            self.rknn
                .input_set_slice(idx, data, false, RknnTensorType::Float32, RknnTensorFormat::Undefined)
                .map_err(|e| anyhow::anyhow!("input_set_slice f32[{}] failed: {}", idx, e))?;
        }
        self.rknn.run().map_err(|e| anyhow::anyhow!("rknn_run failed: {}", e))?;
        let out = self.rknn.outputs_get::<f32>().map_err(|e| anyhow::anyhow!("outputs_get failed: {}", e))?;
        Ok(out.to_vec())
    }
}

/// UNet model that uses rknn-sys directly to set all 4 inputs in one call.
///
/// Model tensor attrs (from rknn_query):
///   [0] sample [1,64,64,4] NHWC Float16
///   [1] timestep [1] Int64
///   [2] encoder_hidden_states [1,77,768] Float16
///   [3] timestep_cond [1,256] Float16
/// Output:
///   [0] out_sample [1,4,64,64] NCHW Float16
pub struct UNetModel {
    ctx: sys::rknn_context,
}

impl Drop for UNetModel {
    fn drop(&mut self) {
        if self.ctx != 0 {
            unsafe { sys::rknn_destroy(self.ctx) };
        }
    }
}

impl UNetModel {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy();
        let path_cstr = std::ffi::CString::new(path_str.as_ref())
            .context("Invalid model path")?;

        let mut ctx: sys::rknn_context = 0;
        let ret = unsafe {
            sys::rknn_init(&mut ctx, path_cstr.as_ptr() as *mut std::ffi::c_void, 0, 0, null_mut())
        };
        if ret != 0 {
            anyhow::bail!("rknn_init failed: {}", ret);
        }

        // The UNet is 1.7GB; enable all 3 NPU cores to ensure enough NPU memory
        let ret = unsafe {
            sys::rknn_set_core_mask(ctx, sys::_rknn_core_mask_RKNN_NPU_CORE_0_1_2)
        };
        if ret != 0 {
            eprintln!("  [warn] rknn_set_core_mask failed: {} (continuing)", ret);
        }
        eprintln!("  UNet: using all 3 NPU cores");
        Ok(Self { ctx })
    }

    pub fn print_info(&self) -> Result<()> {
        // Query io num
        let mut io_num = sys::_rknn_input_output_num { n_input: 0, n_output: 0 };
        unsafe {
            sys::rknn_query(
                self.ctx,
                sys::_rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM,
                &mut io_num as *mut _ as *mut std::ffi::c_void,
                std::mem::size_of::<sys::_rknn_input_output_num>() as u32,
            );
        }
        eprintln!("UNet io_num: inputs={}, outputs={}", io_num.n_input, io_num.n_output);
        Ok(())
    }

    /// Run UNet with all inputs set in a single rknn_inputs_set call.
    pub fn run(
        &self,
        sample_nhwc: &[f32],  // [1, 64, 64, 4]
        timestep: i64,
        text_emb: &[f32],     // [1, 77, 768]
        ts_cond: &[f32],      // [1, 256]
    ) -> Result<Vec<f32>> {
        let timestep_arr = [timestep];

        // Build 4 rknn_input structs
        let mut inputs = [
            sys::_rknn_input {
                index: 0,
                buf: sample_nhwc.as_ptr() as *mut std::ffi::c_void,
                size: (sample_nhwc.len() * 4) as u32, // float32
                pass_through: 0,
                type_: sys::_rknn_tensor_type_RKNN_TENSOR_FLOAT32,
                fmt: sys::_rknn_tensor_format_RKNN_TENSOR_NHWC,
            },
            sys::_rknn_input {
                index: 1,
                buf: timestep_arr.as_ptr() as *mut std::ffi::c_void,
                size: 8, // 1 * sizeof(int64)
                pass_through: 0,
                type_: sys::_rknn_tensor_type_RKNN_TENSOR_INT64,
                fmt: sys::_rknn_tensor_format_RKNN_TENSOR_UNDEFINED,
            },
            sys::_rknn_input {
                index: 2,
                buf: text_emb.as_ptr() as *mut std::ffi::c_void,
                size: (text_emb.len() * 4) as u32,
                pass_through: 0,
                type_: sys::_rknn_tensor_type_RKNN_TENSOR_FLOAT32,
                fmt: sys::_rknn_tensor_format_RKNN_TENSOR_UNDEFINED,
            },
            sys::_rknn_input {
                index: 3,
                buf: ts_cond.as_ptr() as *mut std::ffi::c_void,
                size: (ts_cond.len() * 4) as u32,
                pass_through: 0,
                type_: sys::_rknn_tensor_type_RKNN_TENSOR_FLOAT32,
                fmt: sys::_rknn_tensor_format_RKNN_TENSOR_UNDEFINED,
            },
        ];

        let ret = unsafe { sys::rknn_inputs_set(self.ctx, 4, inputs.as_mut_ptr()) };
        if ret != 0 {
            anyhow::bail!("rknn_inputs_set (UNet) failed: {}", ret);
        }

        let ret = unsafe { sys::rknn_run(self.ctx, null_mut()) };
        if ret != 0 {
            anyhow::bail!("rknn_run (UNet) failed: {}", ret);
        }
        let mut out = sys::_rknn_output {
            want_float: 1,
            is_prealloc: 0,
            index: 0,
            buf: null_mut(),
            size: 0,
        };
        let ret = unsafe { sys::rknn_outputs_get(self.ctx, 1, &mut out, null_mut()) };
        if ret != 0 {
            anyhow::bail!("rknn_outputs_get (UNet) failed: {}", ret);
        }

        let n_elems = out.size as usize / 4;
        let slice = unsafe { std::slice::from_raw_parts(out.buf as *const f32, n_elems) };
        let result = slice.to_vec();

        unsafe { sys::rknn_outputs_release(self.ctx, 1, &mut out) };

        Ok(result)
    }
}
