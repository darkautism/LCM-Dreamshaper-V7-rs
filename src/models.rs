use anyhow::{Context, Result};
use rknn_rs::prelude::{Rknn, RknnCoreMask, RknnTensorFormat, RknnTensorType};
use std::path::Path;
pub struct RknnModel {
    rknn: Rknn,
}

impl RknnModel {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let rknn = Rknn::new(path).context("rknn_init failed")?;
        Ok(Self { rknn })
    }

    pub fn print_info(&self) -> Result<()> {
        let io = self.rknn.io_num().context("io_num query failed")?;
        eprintln!("  io_num: inputs={}, outputs={}", io.n_input, io.n_output);
        Ok(())
    }

    fn run_and_collect(&self) -> Result<Vec<f32>> {
        self.rknn.run().context("rknn_run failed")?;
        let out = self.rknn.outputs_get::<f32>().context("rknn_outputs_get failed")?;
        Ok(out.to_vec())
    }

    pub fn run_f32(&self, inputs: &[(usize, &[f32])]) -> Result<Vec<f32>> {
        for &(idx, data) in inputs {
            self.rknn
                .input_set_slice(idx, data, false, RknnTensorType::Float32, RknnTensorFormat::NHWC)
                .with_context(|| format!("input_set_slice[{}] failed", idx))?;
        }
        self.run_and_collect()
    }

    pub fn run_with_int32_inputs(
        &self,
        int_inputs: &[(usize, &[i32])],
        f32_inputs: &[(usize, &[f32])],
    ) -> Result<Vec<f32>> {
        for &(idx, data) in int_inputs {
            self.rknn
                .input_set_slice(idx, data, false, RknnTensorType::Int32, RknnTensorFormat::Undefined)
                .with_context(|| format!("input_set_slice int32[{}] failed", idx))?;
        }
        for &(idx, data) in f32_inputs {
            self.rknn
                .input_set_slice(idx, data, false, RknnTensorType::Float32, RknnTensorFormat::Undefined)
                .with_context(|| format!("input_set_slice f32[{}] failed", idx))?;
        }
        self.run_and_collect()
    }
}

pub struct UNetModel {
    rknn: Rknn,
}

impl UNetModel {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let rknn = Rknn::new(path).context("rknn_init UNet failed")?;
        rknn.set_core_mask(RknnCoreMask::Core0_1_2)
            .map_err(|e| anyhow::anyhow!("set_core_mask failed: {}", e))?;
        eprintln!("  UNet: using all 3 NPU cores");
        Ok(Self { rknn })
    }

    pub fn print_info(&self) -> Result<()> {
        let io = self.rknn.io_num().context("io_num query failed")?;
        eprintln!("UNet io_num: inputs={}, outputs={}", io.n_input, io.n_output);
        if let Ok(attrs) = self.rknn.input_attrs() {
            for (i, a) in attrs.iter().enumerate() {
                eprintln!("  input[{}]: name={} dims={:?} type={:?} fmt={:?} size={}",
                    i, a.name, a.dims, a.type_, a.fmt, a.size);
            }
        }
        Ok(())
    }

    pub fn run(&self, sample_nhwc: &[f32], timestep: i64, text_emb: &[f32], ts_cond: &[f32]) -> Result<Vec<f32>> {
        // Must set all inputs in a single rknn_inputs_set call —
        // the RKNN 2.3.x runtime misbehaves when inputs are set one-by-one with n=1.
        self.rknn.inputs_set_batch(&[
            (0, bytemuck::cast_slice(sample_nhwc), false, RknnTensorType::Float32, RknnTensorFormat::NHWC),
            (1, bytemuck::bytes_of(&timestep),     false, RknnTensorType::Int64,   RknnTensorFormat::Undefined),
            (2, bytemuck::cast_slice(text_emb),    false, RknnTensorType::Float32, RknnTensorFormat::Undefined),
            (3, bytemuck::cast_slice(ts_cond),     false, RknnTensorType::Float32, RknnTensorFormat::Undefined),
        ]).context("UNet inputs_set_batch failed")?;
        self.rknn.run().context("rknn_run UNet failed")?;
        let out = self.rknn.outputs_get::<f32>().context("rknn_outputs_get UNet failed")?;
        Ok(out.to_vec())
    }
}
