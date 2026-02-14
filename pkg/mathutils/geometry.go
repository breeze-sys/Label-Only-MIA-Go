package mathutils

import (
	"math"
)

// Interpolate 执行线性插值 (Linear Interpolation)。
// 对应 Python: binary_search 中的中间点计算
// 公式: result = a + (b - a) * t
// 用于在“原图”和“对抗样本”之间寻找刚好能骗过模型的那个边界点。
func Interpolate(a, b []float32, t float32) []float32 {
	if len(a) != len(b) {
		panic("mathutils.Interpolate: 输入向量长度不一致")
	}

	result := make([]float32, len(a))
	for i := range a {
		// 逻辑: 起点 + 差距 * 比例
		result[i] = a[i] + (b[i]-a[i])*t
	}
	return result
}

// Normalize 将向量归一化为单位向量 (Unit Vector)。
// 对应 Python: v / np.linalg.norm(v)
// 用于将估算出来的梯度方向标准化。
// 注意：虽然中间计算用 float64 保证精度，但返回仍为 float32 以匹配 API 协议。
func Normalize(v []float32) []float32 {
	norm := L2Norm(v) // L2Norm 返回 float64

	if norm < 1e-12 {
		return make([]float32, len(v))
	}

	result := make([]float32, len(v))
	// 转换回 float32 进行缩放
	scale := 1.0 / float32(norm)

	for i := range v {
		result[i] = v[i] * scale
	}
	return result
}

// ProjectToSphere 将向量投影到 L2 球面上。
// 用于限制扰动大小 (Epsilon)。
// 逻辑: 如果向量模长超过半径，则将其缩放到半径长度。
func ProjectToSphere(v []float32, radius float32) []float32 {
	norm := L2Norm(v)

	// 如果模长小于半径，直接返回副本
	if float32(norm) <= radius {
		dst := make([]float32, len(v))
		copy(dst, v)
		return dst
	}

	// 如果超出半径，进行缩放
	scale := radius / float32(norm)
	result := make([]float32, len(v))
	for i := range v {
		result[i] = v[i] * scale
	}
	return result
}

// CosineSim 计算两个向量的余弦相似度。
// 用于评估梯度估算的准确性。
func CosineSim(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("mathutils.CosineSim: 输入向量长度不一致")
	}

	var dotProduct float64
	var normASq, normBSq float64

	for i := range a {
		valA := float64(a[i])
		valB := float64(b[i])

		dotProduct += valA * valB
		normASq += valA * valA
		normBSq += valB * valB
	}

	if normASq == 0 || normBSq == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normASq) * math.Sqrt(normBSq))
}
