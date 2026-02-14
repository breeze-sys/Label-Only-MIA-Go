package attack

import (
	"math"
	"my-project/pkg/core"
	"my-project/pkg/mathutils"
)

// HSJAConfig 配置攻击参数
type HSJAConfig struct {
	MaxQueries    int     // 最大查询次数限制
	MaxIterations int     // HSJA 的迭代轮数 (默认 50)
	NumEvals      int     // 梯度估计的采样次数 (默认 100)
	InitEvals     int     // 初始化时的采样次数 (默认 100)
	ClipMin       float32 // 0.0
	ClipMax       float32 // 1.0
}

// HSJA 攻击器结构体
type HSJA struct {
	config HSJAConfig
}

// NewHSJA 创建攻击器
func NewHSJA(cfg HSJAConfig) *HSJA {
	if cfg.NumEvals == 0 { cfg.NumEvals = 100 }
	if cfg.MaxIterations == 0 { cfg.MaxIterations = 50 }
	if cfg.InitEvals == 0 { cfg.InitEvals = 100 }
	return &HSJA{config: cfg}
}

// Attack 实现 core.Attacker 接口
func (atk *HSJA) Attack(sample core.Sample, model core.Model) core.AttackResult {
	// 记录查询次数
	queries := 0
	
	// 封装一个带计数的预测函数
	predictFunc := func(img []float32) int {
		queries++
		l, _ := model.Predict(img)
		return l
	}

	original := sample.Data
	targetLabel := sample.Label

	// 1. 初始化：寻找初始对抗样本
	xAdv := atk.initialize(original, targetLabel, predictFunc)
	
	// 如果无法初始化（找不到任何对抗样本），则攻击失败
	if xAdv == nil {
		return core.AttackResult{
			SampleID: sample.ID, OriginalLabel: targetLabel, FinalLabel: targetLabel,
			IsSuccess: false, Queries: queries, Distance: 0.0, IsMember: false, // 距离无法计算
		}
	}

	// 2. 二分查找：找到决策边界
	xAdv = atk.binarySearch(original, xAdv, targetLabel, predictFunc)

	// 3. 迭代优化
	// 计算初始 L2 距离 (注意: L2Distance 返回 float64)
	dist := mathutils.L2Distance(original, xAdv)

	for i := 0; i < atk.config.MaxIterations; i++ {
		// 检查查询次数限制
		if queries >= atk.config.MaxQueries {
			break
		}

		// A. 梯度估计
		delta := atk.computeDelta(float32(dist), i)
		grad := atk.approximateGradient(xAdv, targetLabel, delta, predictFunc)

		// B. 几何级数步进 (Geometric Progression)
		stepSize := atk.computeStepSize(float32(dist), i)
		
		// x_new = x_adv + step_size * grad
		// 修正：使用 VectorScale 和 VectorAdd
		stepVec := mathutils.VectorScale(grad, stepSize)
		xNew := mathutils.VectorAdd(xAdv, stepVec)
		
		// C. 投影与裁剪
		// 投影回合法像素范围 (Box Constraint)
		xNew = mathutils.Clip(xNew, atk.config.ClipMin, atk.config.ClipMax)
		
		// D. 再次二分查找，确保贴紧边界
		xNew = atk.binarySearch(original, xNew, targetLabel, predictFunc)

		// E. 更新最优解
		newDist := mathutils.L2Distance(original, xNew)
		if newDist < dist {
			dist = newDist
			xAdv = xNew
		}
	}

	// 获取最终标签
	finalLabel := predictFunc(xAdv)

	return core.AttackResult{
		SampleID:      sample.ID,
		OriginalLabel: targetLabel,
		FinalLabel:    finalLabel,
		IsSuccess:     finalLabel != targetLabel,
		Queries:       queries,
		Distance:      dist,
		IsMember:      false, // 具体的 Member 判定逻辑通常在 CSV 分析阶段或根据 Threshold 判定
	}
}

// initialize 寻找初始对抗样本
func (atk *HSJA) initialize(original []float32, label int, predict func([]float32) int) []float32 {
	if predict(original) != label {
		return original
	}

	inputSize := len(original)
	for i := 0; i < atk.config.InitEvals; i++ {
		// 修正：假设 GenUniform 在 noise.go 中
		noise := mathutils.GenUniform(inputSize, float64(atk.config.ClipMin), float64(atk.config.ClipMax))
		
		if predict(noise) != label {
			return noise
		}
	}
	return nil
}

// binarySearch 二分查找边界
func (atk *HSJA) binarySearch(original, adversarial []float32, targetLabel int, predict func([]float32) int) []float32 {
	low := 0.0
	high := 1.0
	boundaryPoint := adversarial

	for i := 0; i < 10; i++ {
		mid := (low + high) / 2.0
		
		// mathutils/geometry.go 应包含 Interpolate
		// candidate = original + (adversarial - original) * mid
		// 即: Interpolate(original, adversarial, mid)
		candidate := mathutils.Interpolate(original, adversarial, float32(mid))
		candidate = mathutils.Clip(candidate, atk.config.ClipMin, atk.config.ClipMax)

		if predict(candidate) != targetLabel {
			high = mid
			boundaryPoint = candidate
		} else {
			low = mid
		}
	}
	return boundaryPoint
}

// approximateGradient 梯度估计
func (atk *HSJA) approximateGradient(sample []float32, label int, delta float32, predict func([]float32) int) []float32 {
	numEvals := atk.config.NumEvals
	inputSize := len(sample)
	var validDirections [][]float32

	for j := 0; j < numEvals; j++ {
		// 1. 生成高斯噪声 (noise.go)
		noise := mathutils.GenGaussian(inputSize, 0, 1)
		
		// 2. 归一化 (geometry.go)
		noise = mathutils.Normalize(noise)
		
		// 3. 构造扰动: sample + delta * noise
		perturbation := mathutils.VectorScale(noise, delta)
		posPoint := mathutils.VectorAdd(sample, perturbation)
		posPoint = mathutils.Clip(posPoint, atk.config.ClipMin, atk.config.ClipMax)
		
		// 4. 查询并记录方向
		pred := predict(posPoint)
		if pred != label {
			validDirections = append(validDirections, noise)
		} else {
			// 方向取反: -1 * noise
			validDirections = append(validDirections, mathutils.VectorScale(noise, -1.0))
		}
	}

	// 5. 平均并归一化 (stats.go / geometry.go)
	if len(validDirections) == 0 {
		return mathutils.NewVector(inputSize, 0) // basic.go
	}
	
	// 修正：假设 MeanVector 在 stats.go 中
	grad := mathutils.MeanVector(validDirections)
	return mathutils.Normalize(grad)
}

func (atk *HSJA) computeDelta(dist float32, iter int) float32 {
	if iter == 0 { return 0.1 }
	return dist * 0.1 / float32(math.Sqrt(float64(iter)))
}

func (atk *HSJA) computeStepSize(dist float32, iter int) float32 {
	return dist / float32(math.Sqrt(float64(iter)+1))
}
