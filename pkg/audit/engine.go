package audit

import (
	"Label-Only-MIA-Go/pkg/core"
	"Label-Only-MIA-Go/pkg/mathutils"
	"sync"
)

// 定义信号状态
const (
	SignalRed    = "RED"
	SignalYellow = "YELLOW"
	SignalGreen  = "GREEN"
)

// pkg/audit/engine.go

type AuditThresholds struct {
	Tau95  float64 `json:"tau_95"`
	TauOpt float64 `json:"tau_opt"`
	TauD   float64 `json:"tau_d"`
	TauCV  float64 `json:"tau_cv"`
}

type Engine struct {
	thresholds AuditThresholds
	shadow     core.Model
	target     core.Model
	attacker   core.Attacker
}

func NewEngine(t AuditThresholds, s, tg core.Model, atk core.Attacker) *Engine {
	return &Engine{thresholds: t, shadow: s, target: tg, attacker: atk}
}

func (e *Engine) AuditSample(sample core.Sample) core.AuditResult {
	res := core.AuditResult{
		SampleID:     sample.ID,
		Label:        sample.Label,
		IsMemberTrue: sample.IsMember,
	}

	// ---------------------------------------------------------
	// 1. 预校验：探测目标模型对原图的初始反应
	// ---------------------------------------------------------
	tmpOrig := core.Sample{Data: sample.Data, Label: sample.Label}
	origAtk := e.attacker.Attack(tmpOrig, e.target)

	// 【核心修复 A】：如果模型预测错误，直接判绿并退出
	if origAtk.Distance < 1e-6 {
		res.ShadowLoss = 23.0
		res.MeanDistance = 0
		res.VolatilityCV = 99.0
		res.Conclusion = "🟢 【 安全样本 - 非成员 (模型预测错误) 】"
		return res
	}

	// ---------------------------------------------------------
	// 2. 方案一信号：影子模型行为指纹 (迁移攻击)
	// ---------------------------------------------------------
	// 【核心修复 B】：必须先拿到目标模型的预测结果作为 TargetLabel
	// 之前的代码里 TargetLabel 是空的，导致 CrossEntropy 永远算不对
	targetPred, _ := e.target.Predict(sample.Data)
	sample.TargetLabel = targetPred

	logits, _ := e.shadow.PredictLogits(sample.Data)
	probs := mathutils.Softmax(logits)
	loss := mathutils.CrossEntropy(probs, sample.TargetLabel)
	res.ShadowLoss = loss

	s1 := SignalGreen
	if loss < e.thresholds.Tau95 {
		s1 = SignalRed
	} else if loss < e.thresholds.TauOpt {
		s1 = SignalYellow
	}

	// ---------------------------------------------------------
	// 3. 方案二信号：目标模型几何指纹 (边界攻击)
	// ---------------------------------------------------------
	variants := mathutils.GenerateVariants(sample.Data, 0.001, 10)
	otherDists := e.probeAll(variants, sample.Label)
	allDists := append([]float64{origAtk.Distance}, otherDists...)

	dBar, std := mathutils.MeanAndStd(allDists)
	res.MeanDistance = dBar
	cv := 0.0
	if dBar > 0 {
		cv = std / dBar
	}
	res.VolatilityCV = cv

	s2 := SignalGreen
	// 【核心优化】：只要距离比红线远，且波动比绿线小
	if dBar > e.thresholds.TauD && cv < e.thresholds.TauCV {
		s2 = SignalRed
	} else if dBar > e.thresholds.TauD || cv < e.thresholds.TauCV {
		s2 = SignalYellow
	}

	res.Conclusion = e.fusionLogic(s1, s2)
	return res
}

// 辅助函数微调：去掉原图合并，只跑变体
func (e *Engine) probeAll(variants [][]float32, label int) []float64 {
	dists := make([]float64, len(variants))
	var wg sync.WaitGroup
	for i, img := range variants {
		wg.Add(1)
		go func(idx int, targetImg []float32) {
			defer wg.Done()
			tmp := core.Sample{Data: targetImg, Label: label}
			atkRes := e.attacker.Attack(tmp, e.target)
			dists[idx] = atkRes.Distance
		}(i, img)
	}
	wg.Wait()
	return dists
}

func (e *Engine) fusionLogic(s1, s2 string) string {
	if s1 == SignalRed && s2 == SignalRed {
		return "🔴 【 确认为模型成员 】"
	}
	if s1 == SignalRed || s2 == SignalRed || (s1 == SignalYellow && s2 == SignalYellow) {
		return "🟡 【 风险极高 - 高度可疑 】"
	}
	if s1 == SignalYellow || s2 == SignalYellow {
		return "🟠 【 中度风险 - 疑似成员 】"
	}
	return "🟢 【 安全样本 - 非成员 】"
}
