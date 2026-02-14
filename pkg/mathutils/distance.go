package mathutils

import (
	"math"
)

// ============================================================================
// 距离度量工具库 (Distance Metrics)
// 对应 Python 库: foolbox.distances, numpy.linalg
// ============================================================================

// L2Distance 计算两个向量之间的欧氏距离 (Euclidean Distance)。
// 对应 Python: np.linalg.norm(a - b)
// 这是 MIA (成员推理攻击) 中判定样本是否属于训练集的核心指标。
// 距离越小，两张图片越相似。
func L2Distance(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("mathutils.L2Distance: 输入向量长度不一致")
	}

	var sum float64
	for i := range a {
		// 使用 float64 进行累加，防止精度丢失或溢出
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}

	return math.Sqrt(sum)
}

// L2Norm 计算单个向量的 L2 范数 (模长)。
// 对应 Python: np.linalg.norm(v)
// 用于计算梯度的长度，或者在归一化向量时使用。
func L2Norm(v []float32) float64 {
	var sum float64
	for _, val := range v {
		fval := float64(val)
		sum += fval * fval
	}
	return math.Sqrt(sum)
}

// LinfDistance 计算两个向量之间的切比雪夫距离 (L-infinity Distance)。
// 对应 Python: np.max(np.abs(a - b))
// 即所有像素点中，差异最大的那个点的差异值。
// 常用于衡量对抗样本在 worst-case 下的扰动程度。
func LinfDistance(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("mathutils.LinfDistance: 输入向量长度不一致")
	}

	var maxDiff float64
	for i := range a {
		diff := math.Abs(float64(a[i] - b[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff
}

// L0Distance 计算两个向量之间的 L0 距离 (Hamming-like)。
// 对应 Python: np.count_nonzero(a != b)
// 统计有多少个像素点发生了变化（不考虑变化的幅度，只考虑是否变化）。
// 注意：由于浮点数精度问题，极小的差异也会被计入。
func L0Distance(a, b []float32) int {
	if len(a) != len(b) {
		panic("mathutils.L0Distance: 输入向量长度不一致")
	}

	count := 0
	for i := range a {
		// 直接比较不相等。
		// 如果需要忽略极微小的浮点误差，可以改为 math.Abs(a[i]-b[i]) > 1e-6
		if a[i] != b[i] {
			count++
		}
	}
	return count
}
