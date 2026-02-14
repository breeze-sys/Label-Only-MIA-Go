package mathutils

import (
	"math"
)

// ============================================================================
// 统计辅助工具库 (Statistical Helpers)
// 对应 Python: numpy (argmax, mean), torch.nn.functional (softmax)
// ============================================================================

// ArgMax 找到切片中最大值的索引 (Index of Maximum Value)。
// 对应 Python: np.argmax(probs)
// 用途: 将模型输出的概率数组转换为具体的类别标签 (Label)。
// 例如: [0.1, 0.8, 0.1] -> 返回 1
func ArgMax(probs []float32) int {
	if len(probs) == 0 {
		return -1 // 或者 panic，视具体需求而定
	}

	maxIndex := 0
	maxValue := probs[0]

	for i, v := range probs {
		if v > maxValue {
			maxValue = v
			maxIndex = i
		}
	}
	return maxIndex
}

// MeanVector 计算一组向量的平均值向量。
// 对应 Python: np.mean(vectors, axis=0)
// 输入: 一个包含 n 个向量的切片 (Batch)
// 输出: 一个平均向量
// 用途: HopSkipJump 算法中，需要对多次随机扰动后的梯度估算值取平均，以消除噪声。
func MeanVector(vectors [][]float32) []float32 {
	if len(vectors) == 0 {
		return nil
	}

	rows := len(vectors)
	cols := len(vectors[0])

	// 使用 float64 累加器以防止精度丢失或溢出
	sum := make([]float64, cols)

	for _, vec := range vectors {
		if len(vec) != cols {
			panic("mathutils.MeanVector: 输入向量长度不一致")
		}
		for i, val := range vec {
			sum[i] += float64(val)
		}
	}

	// 计算平均值并转回 float32
	result := make([]float32, cols)
	div := float64(rows)
	for i := 0; i < cols; i++ {
		result[i] = float32(sum[i] / div)
	}

	return result
}

// Softmax 将 Logits (未归一化的分数) 转换为概率分布。
// 对应 Python: torch.nn.functional.softmax(dim=1)
// 公式: P_i = exp(x_i) / sum(exp(x_j))
// 优化: 实现了 "Numerical Stability" (减去最大值)，防止 exp 计算溢出。
func Softmax(logits []float32) []float32 {
	if len(logits) == 0 {
		return []float32{}
	}

	// 1. 找到最大值 (为了数值稳定性)
	maxLogit := logits[0]
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	// 2. 计算指数并求和
	exps := make([]float64, len(logits))
	var sumExps float64
	for i, v := range logits {
		// 减去 maxLogit 防止溢出 (e^1000 会无穷大，但 e^(1000-1000) = 1)
		exps[i] = math.Exp(float64(v - maxLogit))
		sumExps += exps[i]
	}

	// 3. 归一化
	probs := make([]float32, len(logits))
	for i, v := range exps {
		probs[i] = float32(v / sumExps)
	}

	return probs
}
