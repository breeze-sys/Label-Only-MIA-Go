package mathutils

import (
	"math/rand"
	"sync"
	"time"
)

// ============================================================================
// 随机噪声生成工具库 (Noise Generation)
// 对应 Python 库: numpy.random
// 适用场景: 梯度估算 (HopSkipJump), 初始点生成 (Boundary Attack)
// ============================================================================

var (
	// 创建一个全局的随机数生成器
	// 默认使用当前时间作为种子，确保每次运行结果不同
	rng = rand.New(rand.NewSource(time.Now().UnixNano()))

	// 互斥锁：确保在并发环境下（Member C 的 Worker Pool）生成随机数不会冲突
	rngMutex sync.Mutex
)

// SetSeed 设置随机数种子。
// 对应 Python: np.random.seed(seed)
// 用于复现实验结果。如果设置了相同的种子，生成的噪声序列将完全一致。
func SetSeed(seed int64) {
	rngMutex.Lock()
	defer rngMutex.Unlock()

	rng.Seed(seed)
}

// GenGaussian 生成符合高斯/正态分布 (Gaussian/Normal Distribution) 的随机向量。
// 对应 Python: np.random.normal(loc=mean, scale=std, size=size)
//
// 输入:
//   - size: 向量长度 (即 FlattenedSize, 3072)
//   - mean: 均值 (通常为 0)
//   - std:  标准差 (Standard Deviation)
//
// 输出:
//   - []float32: 填充了随机数的切片
//
// 用途:
//   - HopSkipJump 算法中用于生成随机扰动，以估算梯度方向 (Monte Carlo Method)。
func GenGaussian(size int, mean, std float64) []float32 {
	rngMutex.Lock()
	defer rngMutex.Unlock()

	result := make([]float32, size)
	for i := 0; i < size; i++ {
		// NormFloat64 返回标准正态分布随机数 (mean=0, std=1)
		// 公式转换: X = mean + std * Z
		val := rng.NormFloat64()
		result[i] = float32(mean + std*val)
	}
	return result
}

// GenUniform 生成符合均匀分布 (Uniform Distribution) 的随机向量。
// 对应 Python: np.random.uniform(low=min, high=max, size=size)
//
// 输入:
//   - size: 向量长度
//   - min:  最小值
//   - max:  最大值
//
// 输出:
//   - []float32: 填充了随机数的切片
//
// 用途:
//   - Boundary Attack 初始化时，用于在空间中随机撒点寻找初始对抗样本。
//   - 或者用于生成随机掩码。
func GenUniform(size int, min, max float64) []float32 {
	rngMutex.Lock()
	defer rngMutex.Unlock()

	result := make([]float32, size)
	dist := max - min

	for i := 0; i < size; i++ {
		// Float64 返回 [0.0, 1.0) 之间的随机数
		// 公式转换: X = min + (max - min) * U
		val := rng.Float64()
		result[i] = float32(min + dist*val)
	}
	return result
}
