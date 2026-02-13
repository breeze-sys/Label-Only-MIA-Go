package basic

// NewVector 创建一个指定大小并填充特定值的向量
// 对应 Python: np.zeros, np.full
func NewVector(size int, val float32) []float32 {
	// make 初始化时默认值为 0，如果 val 是 0 直接返回即可
	vec := make([]float32, size)
	if val != 0 {
		for i := range vec {
			vec[i] = val
		}
	}
	return vec
}

// VectorAdd 两个向量对应元素相加
// 对应 Python: a + b (用于叠加噪声)
func VectorAdd(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("VectorAdd: vectors must have the same length")
	}

	result := make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		result[i] = a[i] + b[i]
	}
	return result
}

// VectorSub 两个向量对应元素相减
// 对应 Python: a - b (用于计算差异向量)
func VectorSub(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("VectorSub: vectors must have the same length")
	}

	result := make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		result[i] = a[i] - b[i]
	}
	return result
}

// VectorMul 两个向量对应元素相乘 (Hadamard product)
// 对应 Python: a * b (对应元素相乘)
func VectorMul(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("VectorMul: vectors must have the same length")
	}

	result := make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		result[i] = a[i] * b[i]
	}
	return result
}

// VectorScale 向量与标量相乘
// 对应 Python: v * scalar (梯度缩放、步长控制)
func VectorScale(v []float32, s float32) []float32 {
	result := make([]float32, len(v))
	for i := 0; i < len(v); i++ {
		result[i] = v[i] * s
	}
	return result
}

// Clip 将向量中的每个元素限制在 [min, max] 范围内
// 对应 Python: np.clip (ART 中的 clip_image 核心)
func Clip(v []float32, min, max float32) []float32 {
	result := make([]float32, len(v))
	for i, val := range v {
		if val < min {
			result[i] = min
		} else if val > max {
			result[i] = max
		} else {
			result[i] = val
		}
	}
	return result
}

// Clone 深拷贝一个向量，防止修改原数据
// 对应 Python: v.copy()
func Clone(v []float32) []float32 {
	// 创建等长的新切片
	result := make([]float32, len(v))
	// 使用内置 copy 函数进行深拷贝
	copy(result, v)
	return result
}
