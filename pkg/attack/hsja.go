package attack

import (
	"Label-Only-MIA-Go/pkg/core"
	"Label-Only-MIA-Go/pkg/mathutils"
	"math"
)

type HSJAConfig struct {
	MaxQueries    int
	MaxIterations int
	NumEvals      int
	InitEvals     int
	ClipMin       float32
	ClipMax       float32
}

type HSJA struct {
	config HSJAConfig
}

func NewHSJA(cfg HSJAConfig) *HSJA {
	if cfg.MaxQueries == 0 {
		cfg.MaxQueries = 10000
	}
	if cfg.NumEvals == 0 {
		cfg.NumEvals = 100
	}
	if cfg.MaxIterations == 0 {
		cfg.MaxIterations = 50
	}
	if cfg.InitEvals == 0 {
		cfg.InitEvals = 100
	}
	if cfg.ClipMax == 0 {
		cfg.ClipMax = 1.0
	}
	return &HSJA{config: cfg}
}

// Attack 实现升级后的 core.Attacker 接口
func (atk *HSJA) Attack(sample core.Sample, model core.Model) core.AttackResult {
	queries := 0

	predictFunc := func(img []float32) int {
		queries++
		l, err := model.Predict(core.Image(img))
		if err != nil {
			return -1
		}
		return l
	}

	original := sample.Data
	targetLabel := predictFunc(original)
	if targetLabel < 0 {
		return core.AttackResult{
			SampleID: sample.ID,
			Distance: 0.0,
			Queries:  queries,
		}
	}

	// 1. 初始化
	xAdv := atk.initialize(original, targetLabel, predictFunc)
	if xAdv == nil {
		return core.AttackResult{
			SampleID: sample.ID,
			Distance: 0.0,
			Queries:  queries,
		}
	}

	// 2. 初始二分查找
	xAdv = atk.binarySearch(original, xAdv, targetLabel, predictFunc)
	dist := mathutils.L2Distance(original, xAdv)

	// 3. 迭代优化循环
	for i := 0; i < atk.config.MaxIterations; i++ {
		if queries >= atk.config.MaxQueries {
			break
		}

		// A. 梯度估计
		delta := atk.computeDelta(float32(dist), i)
		grad := atk.approximateGradient(xAdv, targetLabel, delta, model, &queries)

		// B. 几何级数步进
		stepSize := atk.computeStepSize(float32(dist), i)
		stepVec := mathutils.VectorScale(grad, stepSize)
		xNew := mathutils.VectorAdd(xAdv, stepVec)

		// C. 投影与裁剪
		xNew = mathutils.Clip(xNew, atk.config.ClipMin, atk.config.ClipMax)

		// D. 再次二分查找
		xNew = atk.binarySearch(original, xNew, targetLabel, predictFunc)

		// E. 更新最优解
		newDist := mathutils.L2Distance(original, xNew)
		if newDist < dist {
			dist = newDist
			xAdv = xNew
		}
	}

	// 最终只返回最核心的物理指标
	return core.AttackResult{
		SampleID: sample.ID,
		Distance: dist,
		Queries:  queries,
	}
}

// initialize 寻找初始对抗样本
func (atk *HSJA) initialize(original []float32, label int, predict func([]float32) int) []float32 {
	if predict(original) != label {
		return original
	}
	inputSize := len(original)
	for i := 0; i < atk.config.InitEvals; i++ {
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

	for i := 0; i < 15; i++ {
		mid := (low + high) / 2.0
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

// approximateGradient 梯度估计 (修正了类型转换问题)
func (atk *HSJA) approximateGradient(sample []float32, label int, delta float32, model core.Model, queries *int) []float32 {
	numEvals := atk.config.NumEvals
	inputSize := len(sample)

	batchImgs := make([]core.Image, numEvals)
	noises := make([][]float32, numEvals)

	for j := 0; j < numEvals; j++ {
		noise := mathutils.GenGaussian(inputSize, 0, 1)
		noise = mathutils.Normalize(noise)
		noises[j] = noise

		perturbation := mathutils.VectorScale(noise, delta)
		posPoint := mathutils.VectorAdd(sample, perturbation)
		posPoint = mathutils.Clip(posPoint, atk.config.ClipMin, atk.config.ClipMax)

		// 显式转换为 core.Image
		batchImgs[j] = core.Image(posPoint)
	}

	preds, err := model.PredictBatch(batchImgs)
	if err != nil {
		return mathutils.NewVector(inputSize, 0)
	}
	*queries += numEvals

	var validDirections [][]float32
	for j, pred := range preds {
		if pred != label {
			validDirections = append(validDirections, noises[j])
		} else {
			validDirections = append(validDirections, mathutils.VectorScale(noises[j], -1.0))
		}
	}

	if len(validDirections) == 0 {
		return mathutils.NewVector(inputSize, 0)
	}
	grad := mathutils.MeanVector(validDirections)
	return mathutils.Normalize(grad)
}

func (atk *HSJA) computeDelta(dist float32, iter int) float32 {
	if iter == 0 {
		return 0.1
	}
	return dist * 0.1 / float32(math.Sqrt(float64(iter)))
}

func (atk *HSJA) computeStepSize(dist float32, iter int) float32 {
	return dist / float32(math.Sqrt(float64(iter)+1))
}
