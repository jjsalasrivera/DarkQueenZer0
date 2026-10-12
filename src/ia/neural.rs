//! Policy/value neural network (AlphaZero) implemented with `candle`.
//!
//! Input: board encoding (see `encoding`).
//! Outputs: policy logits over the 4096 possible actions and a scalar value in
//! [-1, 1] from the perspective of the player to move.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};

use crate::ia::encoding::{INPUT_SIZE, NUM_ACTIONS};

const HIDDEN: usize = 256;

pub struct PolicyValueNet {
    l1: Linear,
    l2: Linear,
    policy: Linear,
    value: Linear,
}

impl PolicyValueNet {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            l1: linear(INPUT_SIZE, HIDDEN, vs.pp("l1"))?,
            l2: linear(HIDDEN, HIDDEN, vs.pp("l2"))?,
            policy: linear(HIDDEN, NUM_ACTIONS, vs.pp("policy"))?,
            value: linear(HIDDEN, 1, vs.pp("value"))?,
        })
    }

    /// Runs a batch of positions `x` shaped `[batch, INPUT_SIZE]`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.l1.forward(x)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        let policy = self.policy.forward(&x)?;
        let value = self.value.forward(&x)?.tanh()?;
        Ok((policy, value))
    }

    /// Evaluates a single encoded position and returns `(policy_logits, value)`.
    pub fn evaluate(&self, encoded: &[f32]) -> Result<(Vec<f32>, f32)> {
        let device = Device::Cpu;
        let x = Tensor::from_slice(encoded, (1, INPUT_SIZE), &device)?;
        let (policy, value) = self.forward(&x)?;

        let policy = policy.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let value = value.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?;
        Ok((policy, value))
    }
}

/// Creates a network with randomly initialized weights.
pub fn random() -> Result<PolicyValueNet> {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    PolicyValueNet::new(vs)
}

/// Builds the network sharing the `varmap` variables (for training).
///
/// By sharing the `VarMap`, the tensors returned by `Linear` share storage with
/// the variables, so optimizer updates (or `VarMap::load`) are reflected without
/// rebuilding the network.
pub fn from_varmap(varmap: &VarMap, device: &Device) -> Result<PolicyValueNet> {
    let vs = VarBuilder::from_varmap(varmap, DType::F32, device);
    PolicyValueNet::new(vs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_shapes_are_correct() {
        let net = random().unwrap();
        let encoded = vec![0.0f32; INPUT_SIZE];
        let (policy, value) = net.evaluate(&encoded).unwrap();
        assert_eq!(policy.len(), NUM_ACTIONS);
        assert!((-1.0..=1.0).contains(&value));
    }
}
