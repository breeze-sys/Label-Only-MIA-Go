package main

import (
	"Label-Only-MIA-Go/pkg/attack"
	"Label-Only-MIA-Go/pkg/audit"
	"Label-Only-MIA-Go/pkg/client"
	"Label-Only-MIA-Go/pkg/core"
	"Label-Only-MIA-Go/pkg/dataset"
	"Label-Only-MIA-Go/pkg/mathutils"
	"Label-Only-MIA-Go/pkg/worker"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type rawShadowConfig struct {
	Tau95          *float64 `json:"tau_95"`
	Tau95Alt       *float64 `json:"Tau95"`
	TauOpt         *float64 `json:"tau_opt"`
	TauOptAlt      *float64 `json:"TauOpt"`
	Threshold      *float64 `json:"threshold"`
	MeanMemberLoss *float64 `json:"mean_member_loss"`
}

type runConfig struct {
	TargetAPI                 string
	ShadowAPI                 string
	ShadowConfigPath          string
	CalibrationDataPath       string
	MemberDataPath            string
	NonMemberDataPath         string
	OutputReportPath          string
	CalibrationCandidateCount int
	CalibrationTargetCount    int
	MinValidStrangers         int
	MemberSampleCount         int
	NonMemberSampleCount      int
	AuditWorkers              int
}

func envOrDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envIntOrDefault(key string, fallback int) int {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		log.Printf("⚠️  环境变量 %s=%q 不是合法整数，回退到默认值 %d", key, value, fallback)
		return fallback
	}
	return parsed
}

func loadRunConfig() runConfig {
	return runConfig{
		TargetAPI:                 envOrDefault("TARGET_API", "http://localhost:8000"),
		ShadowAPI:                 envOrDefault("SHADOW_API", "http://localhost:8001"),
		ShadowConfigPath:          envOrDefault("SHADOW_CONFIG_PATH", "shadow_config.json"),
		CalibrationDataPath:       envOrDefault("CALIBRATION_DATA_PATH", "data/cifar-10-batches-bin/test_batch.bin"),
		MemberDataPath:            envOrDefault("MEMBER_DATA_PATH", "data/cifar-10-batches-bin/data_batch_1.bin"),
		NonMemberDataPath:         envOrDefault("NON_MEMBER_DATA_PATH", "data/cifar-10-batches-bin/test_batch.bin"),
		OutputReportPath:          envOrDefault("OUTPUT_REPORT_PATH", "output/audit_report.json"),
		CalibrationCandidateCount: envIntOrDefault("CALIBRATION_CANDIDATE_COUNT", 100),
		CalibrationTargetCount:    envIntOrDefault("CALIBRATION_TARGET_COUNT", 10),
		MinValidStrangers:         envIntOrDefault("MIN_VALID_STRANGERS", 5),
		MemberSampleCount:         envIntOrDefault("MEMBER_SAMPLE_COUNT", 5),
		NonMemberSampleCount:      envIntOrDefault("NON_MEMBER_SAMPLE_COUNT", 5),
		AuditWorkers:              envIntOrDefault("AUDIT_WORKERS", 20),
	}
}

func loadAuditThresholds(path string) (audit.AuditThresholds, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return audit.AuditThresholds{}, err
	}

	var raw rawShadowConfig
	if err := json.Unmarshal(data, &raw); err != nil {
		return audit.AuditThresholds{}, err
	}

	var thresholds audit.AuditThresholds
	switch {
	case raw.Tau95 != nil:
		thresholds.Tau95 = *raw.Tau95
	case raw.Tau95Alt != nil:
		thresholds.Tau95 = *raw.Tau95Alt
	case raw.Threshold != nil:
		thresholds.Tau95 = *raw.Threshold
	}

	switch {
	case raw.TauOpt != nil:
		thresholds.TauOpt = *raw.TauOpt
	case raw.TauOptAlt != nil:
		thresholds.TauOpt = *raw.TauOptAlt
	case raw.MeanMemberLoss != nil:
		thresholds.TauOpt = *raw.MeanMemberLoss
	case raw.Threshold != nil:
		thresholds.TauOpt = *raw.Threshold
	}

	if thresholds.Tau95 > 0 && thresholds.TauOpt > 0 && thresholds.Tau95 > thresholds.TauOpt {
		thresholds.Tau95, thresholds.TauOpt = thresholds.TauOpt, thresholds.Tau95
	}
	if thresholds.Tau95 == 0 && thresholds.TauOpt > 0 {
		thresholds.Tau95 = thresholds.TauOpt
	}
	if thresholds.TauOpt == 0 && thresholds.Tau95 > 0 {
		thresholds.TauOpt = thresholds.Tau95
	}

	return thresholds, nil
}

func writeAuditReport(path string, reports []core.AuditResult) error {
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(reports, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func main() {
	cfg := loadRunConfig()

	fmt.Println("=====================================================")
	fmt.Println("🛡️  LabelScan-Go: 高性能黑盒模型隐私审计工具 (半发布诊断版)")
	fmt.Println("=====================================================")

	// ---------------------------------------------------------
	// 1. 资产加载 (从 JSON 读取迁移攻击阈值)
	// ---------------------------------------------------------
	thresholds, err := loadAuditThresholds(cfg.ShadowConfigPath)
	if err != nil {
		log.Fatalf("❌ 阈值配置加载失败: %v", err)
	}

	// ---------------------------------------------------------
	// 2. 环境初始化
	// ---------------------------------------------------------
	targetModel := client.NewHTTPClient(cfg.TargetAPI)
	shadowModel := client.NewHTTPClient(cfg.ShadowAPI)

	hsja := attack.NewHSJA(attack.HSJAConfig{
		MaxQueries:    5000,
		MaxIterations: 40,
		NumEvals:      100,
	})

	// ---------------------------------------------------------
	// 3. 现场定标 (Calibration)：核心修复逻辑
	// ---------------------------------------------------------
	fmt.Printf("\n🔍 阶段一：正在进行现场定标 (寻找 %d 个有效路人)...\n", cfg.CalibrationTargetCount)
	loader := &dataset.CifarLoader{}

	// 【关键修改 1】：加载 100 张备选路人图，防止 10 张不够挑导致的死循环
	candidates, err := loader.GetRandomStrangers(cfg.CalibrationDataPath, cfg.CalibrationCandidateCount)
	if err != nil {
		log.Fatalf("❌ 读取定标样本失败: %v", err)
	}

	var refDists [][]float64
	validStrangers := 0

	for i := 0; i < len(candidates) && validStrangers < cfg.CalibrationTargetCount; i++ {
		s := candidates[i]

		// 预探测：模型必须能认对这张图 (Dist > 0)
		tmpOrig := core.Sample{Data: s.Data, Label: s.Label}
		resOrig := hsja.Attack(tmpOrig, targetModel)

		if resOrig.Distance < 1e-5 {
			fmt.Printf("   [跳过] 路人 #%d 预测错误 (Dist=0)，尝试下一个...\n", i+1)
			continue
		}

		fmt.Printf("   [定标中] 正在探测有效路人 %d/%d 的地理特征...\n", validStrangers+1, cfg.CalibrationTargetCount)

		// 生成变体并测距
		variants := mathutils.GenerateVariants(s.Data, 0.001, 10)
		points := append([][]float32{s.Data}, variants...)
		var groupDists []float64
		for _, img := range points {
			tmp := core.Sample{Data: img, Label: s.Label}
			res := hsja.Attack(tmp, targetModel)
			groupDists = append(groupDists, res.Distance)
		}
		refDists = append(refDists, groupDists)
		validStrangers++
	}

	if validStrangers < cfg.MinValidStrangers {
		log.Fatal("❌ 严重错误：无法找到足够的有效路人样本，请检查模型准确率或数据对齐！")
	}

	// 调用统计函数算出 TauD 和 TauCV
	thresholds.TauD, thresholds.TauCV = mathutils.CalibrateReference(refDists)

	// 【关键修改 2】：强制诊断打印
	fmt.Println("\n--- 🕵️ 阈值诊断报告 (核心排查数据) ---")
	fmt.Printf("👉 迁移红灯 (Tau95): %.4f (来自JSON)\n", thresholds.Tau95)
	fmt.Printf("👉 迁移黄灯 (TauOpt): %.4f (来自JSON)\n", thresholds.TauOpt)
	fmt.Printf("👉 距离红线 (TauD):   %.4f (正常应在 0.3-0.7)\n", thresholds.TauD)
	fmt.Printf("👉 波动绿线 (TauCV):  %.4f (正常应在 0.01-0.1)\n", thresholds.TauCV)
	if thresholds.TauD > 1.5 {
		fmt.Println("⚠️  警告：TauD 过高，会导致严重漏报 (Recall低)！")
	}
	fmt.Println("-------------------------------------------")
	fmt.Println()

	// ---------------------------------------------------------
	// 4. 构造混合测试包 (各拿 5 个，总计 10 个样本做快速诊断)
	// ---------------------------------------------------------
	fmt.Printf("📦 阶段二：构造混合测试包 (%d 成员 + %d 路人)...\n", cfg.MemberSampleCount, cfg.NonMemberSampleCount)
	loaderM := &dataset.CifarLoader{IsMemberSet: true}
	members, err := loaderM.LoadBatch(cfg.MemberDataPath, cfg.MemberSampleCount)
	if err != nil {
		log.Fatalf("❌ 读取成员样本失败: %v", err)
	}

	loaderNM := &dataset.CifarLoader{IsMemberSet: false}
	nonMembers, err := loaderNM.LoadBatch(cfg.NonMemberDataPath, cfg.NonMemberSampleCount)
	if err != nil {
		log.Fatalf("❌ 读取非成员样本失败: %v", err)
	}

	targetSamples := append(members, nonMembers...)
	for i := range targetSamples {
		targetSamples[i].ID = i
	}

	// ---------------------------------------------------------
	// 5. 并发审计流水线
	// ---------------------------------------------------------
	engine := audit.NewEngine(thresholds, shadowModel, targetModel, hsja)
	pool := worker.NewAuditPool(engine, cfg.AuditWorkers)

	fmt.Println("🚀 阶段三：全自动化审计流水线开启...")
	finalReports := pool.RunAudit(targetSamples)

	// ---------------------------------------------------------
	// 6. 战报评估
	// ---------------------------------------------------------
	fmt.Println("\n=====================================================")
	fmt.Println("📈 LabelScan-Go 最终审计效能战报")
	fmt.Println("-----------------------------------------------------")

	var tp, fp, tn, fn int
	for _, r := range finalReports {
		predIsMember := strings.Contains(r.Conclusion, "🔴") ||
			strings.Contains(r.Conclusion, "🟡") ||
			strings.Contains(r.Conclusion, "🟠")

		if predIsMember == r.IsMemberTrue {
			if r.IsMemberTrue {
				tp++
			} else {
				tn++
			}
		} else {
			if r.IsMemberTrue {
				fn++
			} else {
				fp++
			}
		}
	}

	total := len(finalReports)
	accuracy := float64(tp+tn) / float64(total) * 100
	precision := 0.0
	if (tp + fp) > 0 {
		precision = float64(tp) / float64(tp+fp) * 100
	}
	recall := 0.0
	if (tp + fn) > 0 {
		recall = float64(tp) / float64(tp+fn) * 100
	}

	fmt.Printf("   > 总审计样本数：   %d\n", total)
	fmt.Printf("   > 审计准确率 (ACC): %.2f%%\n", accuracy)
	fmt.Printf("   > 查准率 (Precision): %.2f%%\n", precision)
	fmt.Printf("   > 查全率 (Recall):    %.2f%%\n", recall)
	fmt.Println("=====================================================")

	if err := writeAuditReport(cfg.OutputReportPath, finalReports); err != nil {
		log.Printf("⚠️  保存审计报告失败: %v", err)
	} else {
		fmt.Printf("💾 详细审计报告已写入 %s\n", cfg.OutputReportPath)
	}
}
