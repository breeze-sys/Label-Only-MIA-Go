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
	"flag"
	"fmt"
	"html/template"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type runConfig struct {
	Preset                    string `json:"preset"`
	TargetAPI                 string `json:"target_api"`
	ShadowAPI                 string `json:"shadow_api"`
	ShadowConfigPath          string `json:"shadow_config_path"`
	CalibrationDataPath       string `json:"calibration_data_path"`
	MemberDataPath            string `json:"member_data_path"`
	NonMemberDataPath         string `json:"non_member_data_path"`
	JSONReportPath            string `json:"json_report_path"`
	HTMLReportPath            string `json:"html_report_path"`
	CalibrationCandidateCount int    `json:"calibration_candidate_count"`
	CalibrationTargetCount    int    `json:"calibration_target_count"`
	MinValidStrangers         int    `json:"min_valid_strangers"`
	MemberSampleCount         int    `json:"member_sample_count"`
	NonMemberSampleCount      int    `json:"non_member_sample_count"`
	AuditWorkers              int    `json:"audit_workers"`
	MaxQueries                int    `json:"max_queries"`
	MaxIterations             int    `json:"max_iterations"`
	NumEvals                  int    `json:"num_evals"`
}

type metricSummary struct {
	Total          int     `json:"total"`
	MemberSamples  int     `json:"member_samples"`
	NonMembers     int     `json:"non_member_samples"`
	PredictedRisk  int     `json:"predicted_risk"`
	TruePositive   int     `json:"true_positive"`
	FalsePositive  int     `json:"false_positive"`
	TrueNegative   int     `json:"true_negative"`
	FalseNegative  int     `json:"false_negative"`
	Accuracy       float64 `json:"accuracy"`
	Precision      float64 `json:"precision"`
	Recall         float64 `json:"recall"`
	HighRiskRate   float64 `json:"high_risk_rate"`
	MeanShadowLoss float64 `json:"mean_shadow_loss"`
	MeanDistance   float64 `json:"mean_boundary_distance"`
	MeanVolatility float64 `json:"mean_volatility_cv"`
}

type reportSample struct {
	SampleID      int     `json:"sample_id"`
	Label         int     `json:"label"`
	IsMemberTrue  bool    `json:"is_member_true"`
	PredictedRisk bool    `json:"predicted_risk"`
	RiskLevel     string  `json:"risk_level"`
	RiskClass     string  `json:"risk_class"`
	ShadowLoss    float64 `json:"shadow_loss"`
	MeanDistance  float64 `json:"mean_boundary_distance"`
	VolatilityCV  float64 `json:"volatility_cv"`
	Conclusion    string  `json:"conclusion"`
}

type auditReport struct {
	Tool        string                `json:"tool"`
	Version     string                `json:"version"`
	GeneratedAt string                `json:"generated_at"`
	Config      runConfig             `json:"config"`
	Thresholds  audit.AuditThresholds `json:"thresholds"`
	Metrics     metricSummary         `json:"metrics"`
	Samples     []reportSample        `json:"samples"`
}

func defaultConfig() runConfig {
	return runConfig{
		Preset:                    "standard",
		TargetAPI:                 "http://localhost:8000",
		ShadowAPI:                 "http://localhost:8001",
		ShadowConfigPath:          "shadow_config.json",
		CalibrationDataPath:       "data/cifar-10-batches-bin/test_batch.bin",
		MemberDataPath:            "data/cifar-10-batches-bin/data_batch_1.bin",
		NonMemberDataPath:         "data/cifar-10-batches-bin/test_batch.bin",
		JSONReportPath:            "output/audit_report.json",
		HTMLReportPath:            "output/audit_report.html",
		CalibrationCandidateCount: 100,
		CalibrationTargetCount:    10,
		MinValidStrangers:         5,
		MemberSampleCount:         5,
		NonMemberSampleCount:      5,
		AuditWorkers:              20,
		MaxQueries:                5000,
		MaxIterations:             40,
		NumEvals:                  100,
	}
}

func applyPreset(cfg *runConfig, preset string) {
	cfg.Preset = preset
	switch preset {
	case "smoke":
		cfg.CalibrationCandidateCount = 10
		cfg.CalibrationTargetCount = 1
		cfg.MinValidStrangers = 1
		cfg.MemberSampleCount = 1
		cfg.NonMemberSampleCount = 1
		cfg.AuditWorkers = 2
		cfg.MaxQueries = 800
		cfg.MaxIterations = 8
		cfg.NumEvals = 30
	case "standard":
		// Keep the conservative defaults used by the project demo.
	case "full":
		cfg.CalibrationCandidateCount = 200
		cfg.CalibrationTargetCount = 20
		cfg.MinValidStrangers = 10
		cfg.MemberSampleCount = 50
		cfg.NonMemberSampleCount = 50
		cfg.AuditWorkers = 20
		cfg.MaxQueries = 5000
		cfg.MaxIterations = 40
		cfg.NumEvals = 100
	case "custom":
		// All values come from flags.
	default:
		log.Fatalf("未知 preset: %s，可选 smoke / standard / full / custom", preset)
	}
}

func loadConfigFromFlags() runConfig {
	cfg := defaultConfig()

	preset := flag.String("preset", cfg.Preset, "运行预设：smoke / standard / full / custom")
	targetAPI := flag.String("target-api", cfg.TargetAPI, "目标模型 HTTP 服务地址")
	shadowAPI := flag.String("shadow-api", cfg.ShadowAPI, "影子模型 HTTP 服务地址")
	shadowConfig := flag.String("shadow-config", cfg.ShadowConfigPath, "影子模型阈值配置路径")
	calibrationPath := flag.String("calibration-data", cfg.CalibrationDataPath, "现场定标数据路径")
	memberPath := flag.String("member-data", cfg.MemberDataPath, "成员样本数据路径")
	nonMemberPath := flag.String("non-member-data", cfg.NonMemberDataPath, "非成员样本数据路径")
	jsonReport := flag.String("json-report", cfg.JSONReportPath, "JSON 报告输出路径")
	htmlReport := flag.String("html-report", cfg.HTMLReportPath, "HTML 报告输出路径，留空则不生成")
	calCandidates := flag.Int("calibration-candidates", cfg.CalibrationCandidateCount, "定标候选样本数")
	calTargets := flag.Int("calibration-targets", cfg.CalibrationTargetCount, "目标有效定标样本数")
	minStrangers := flag.Int("min-valid-strangers", cfg.MinValidStrangers, "最少有效定标样本数")
	memberCount := flag.Int("member-samples", cfg.MemberSampleCount, "成员样本数量")
	nonMemberCount := flag.Int("non-member-samples", cfg.NonMemberSampleCount, "非成员样本数量")
	workers := flag.Int("workers", cfg.AuditWorkers, "并发审计工人数")
	maxQueries := flag.Int("max-queries", cfg.MaxQueries, "HSJA 单样本最大查询次数")
	maxIterations := flag.Int("max-iterations", cfg.MaxIterations, "HSJA 最大迭代轮数")
	numEvals := flag.Int("num-evals", cfg.NumEvals, "HSJA 每轮梯度估计采样数")

	flag.Parse()
	explicit := map[string]bool{}
	flag.Visit(func(f *flag.Flag) { explicit[f.Name] = true })

	applyPreset(&cfg, *preset)

	cfg.TargetAPI = *targetAPI
	cfg.ShadowAPI = *shadowAPI
	cfg.ShadowConfigPath = *shadowConfig
	cfg.CalibrationDataPath = *calibrationPath
	cfg.MemberDataPath = *memberPath
	cfg.NonMemberDataPath = *nonMemberPath
	cfg.JSONReportPath = *jsonReport
	cfg.HTMLReportPath = *htmlReport

	if explicit["calibration-candidates"] {
		cfg.CalibrationCandidateCount = *calCandidates
	}
	if explicit["calibration-targets"] {
		cfg.CalibrationTargetCount = *calTargets
	}
	if explicit["min-valid-strangers"] {
		cfg.MinValidStrangers = *minStrangers
	}
	if explicit["member-samples"] {
		cfg.MemberSampleCount = *memberCount
	}
	if explicit["non-member-samples"] {
		cfg.NonMemberSampleCount = *nonMemberCount
	}
	if explicit["workers"] {
		cfg.AuditWorkers = *workers
	}
	if explicit["max-queries"] {
		cfg.MaxQueries = *maxQueries
	}
	if explicit["max-iterations"] {
		cfg.MaxIterations = *maxIterations
	}
	if explicit["num-evals"] {
		cfg.NumEvals = *numEvals
	}

	if cfg.AuditWorkers < 1 {
		cfg.AuditWorkers = 1
	}
	if cfg.MemberSampleCount < 0 || cfg.NonMemberSampleCount < 0 {
		log.Fatal("样本数量不能为负数")
	}
	return cfg
}

func loadThresholds(path string) (audit.AuditThresholds, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return audit.AuditThresholds{}, err
	}
	var thresholds audit.AuditThresholds
	if err := json.Unmarshal(data, &thresholds); err != nil {
		return audit.AuditThresholds{}, err
	}
	return thresholds, nil
}

func calibrateThresholds(cfg runConfig, thresholds *audit.AuditThresholds, hsja *attack.HSJA, targetModel *client.HTTPClient) error {
	fmt.Printf("\n[1/4] 现场定标：寻找 %d 个有效参考样本\n", cfg.CalibrationTargetCount)
	loader := &dataset.CifarLoader{}
	candidates, err := loader.GetRandomStrangers(cfg.CalibrationDataPath, cfg.CalibrationCandidateCount)
	if err != nil {
		return err
	}

	var refDists [][]float64
	validStrangers := 0
	for i := 0; i < len(candidates) && validStrangers < cfg.CalibrationTargetCount; i++ {
		s := candidates[i]
		resOrig := hsja.Attack(core.Sample{Data: s.Data, Label: s.Label}, targetModel)
		if resOrig.Distance < 1e-5 {
			continue
		}

		variants := mathutils.GenerateVariants(s.Data, 0.001, 10)
		points := append([][]float32{s.Data}, variants...)
		groupDists := make([]float64, 0, len(points))
		for _, img := range points {
			res := hsja.Attack(core.Sample{Data: img, Label: s.Label}, targetModel)
			groupDists = append(groupDists, res.Distance)
		}
		refDists = append(refDists, groupDists)
		validStrangers++
		fmt.Printf("  已完成参考样本 %d/%d\n", validStrangers, cfg.CalibrationTargetCount)
	}

	if validStrangers < cfg.MinValidStrangers {
		return fmt.Errorf("有效参考样本不足：%d < %d", validStrangers, cfg.MinValidStrangers)
	}

	thresholds.TauD, thresholds.TauCV = mathutils.CalibrateReference(refDists)
	return nil
}

func loadAuditSamples(cfg runConfig) ([]core.Sample, error) {
	fmt.Printf("\n[2/4] 加载审计样本：%d 成员 + %d 非成员\n", cfg.MemberSampleCount, cfg.NonMemberSampleCount)

	memberLoader := &dataset.CifarLoader{IsMemberSet: true}
	members, err := memberLoader.LoadBatch(cfg.MemberDataPath, cfg.MemberSampleCount)
	if err != nil {
		return nil, err
	}

	nonMemberLoader := &dataset.CifarLoader{IsMemberSet: false}
	nonMembers, err := nonMemberLoader.LoadBatch(cfg.NonMemberDataPath, cfg.NonMemberSampleCount)
	if err != nil {
		return nil, err
	}

	samples := append(members, nonMembers...)
	for i := range samples {
		samples[i].ID = i
	}
	return samples, nil
}

func isPredictedRisk(conclusion string) bool {
	return strings.Contains(conclusion, "🔴") ||
		strings.Contains(conclusion, "🟡") ||
		strings.Contains(conclusion, "🟠")
}

func riskLevel(conclusion string) (string, string) {
	switch {
	case strings.Contains(conclusion, "🔴"):
		return "confirmed_member", "risk-red"
	case strings.Contains(conclusion, "🟡"):
		return "high_risk", "risk-yellow"
	case strings.Contains(conclusion, "🟠"):
		return "medium_risk", "risk-orange"
	default:
		return "non_member", "risk-green"
	}
}

func buildReport(cfg runConfig, thresholds audit.AuditThresholds, results []core.AuditResult) auditReport {
	sort.Slice(results, func(i, j int) bool { return results[i].SampleID < results[j].SampleID })

	samples := make([]reportSample, 0, len(results))
	var metrics metricSummary
	metrics.Total = len(results)

	for _, r := range results {
		predRisk := isPredictedRisk(r.Conclusion)
		level, className := riskLevel(r.Conclusion)
		if r.IsMemberTrue {
			metrics.MemberSamples++
		} else {
			metrics.NonMembers++
		}
		if predRisk {
			metrics.PredictedRisk++
		}
		switch {
		case predRisk && r.IsMemberTrue:
			metrics.TruePositive++
		case predRisk && !r.IsMemberTrue:
			metrics.FalsePositive++
		case !predRisk && !r.IsMemberTrue:
			metrics.TrueNegative++
		case !predRisk && r.IsMemberTrue:
			metrics.FalseNegative++
		}
		metrics.MeanShadowLoss += r.ShadowLoss
		metrics.MeanDistance += r.MeanDistance
		metrics.MeanVolatility += r.VolatilityCV

		samples = append(samples, reportSample{
			SampleID:      r.SampleID,
			Label:         r.Label,
			IsMemberTrue:  r.IsMemberTrue,
			PredictedRisk: predRisk,
			RiskLevel:     level,
			RiskClass:     className,
			ShadowLoss:    r.ShadowLoss,
			MeanDistance:  r.MeanDistance,
			VolatilityCV:  r.VolatilityCV,
			Conclusion:    r.Conclusion,
		})
	}

	if metrics.Total > 0 {
		total := float64(metrics.Total)
		metrics.Accuracy = float64(metrics.TruePositive+metrics.TrueNegative) / total
		metrics.HighRiskRate = float64(metrics.PredictedRisk) / total
		metrics.MeanShadowLoss /= total
		metrics.MeanDistance /= total
		metrics.MeanVolatility /= total
	}
	if metrics.TruePositive+metrics.FalsePositive > 0 {
		metrics.Precision = float64(metrics.TruePositive) / float64(metrics.TruePositive+metrics.FalsePositive)
	}
	if metrics.TruePositive+metrics.FalseNegative > 0 {
		metrics.Recall = float64(metrics.TruePositive) / float64(metrics.TruePositive+metrics.FalseNegative)
	}

	return auditReport{
		Tool:        "LabelScan-Go",
		Version:     "tool-v1",
		GeneratedAt: time.Now().Format(time.RFC3339),
		Config:      cfg,
		Thresholds:  thresholds,
		Metrics:     metrics,
		Samples:     samples,
	}
}

func writeJSONReport(path string, report auditReport) error {
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func writeHTMLReport(path string, report auditReport) error {
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	tmpl, err := template.New("report").Funcs(template.FuncMap{
		"pct": func(v float64) string { return fmt.Sprintf("%.1f%%", v*100) },
		"num": func(v float64) string { return fmt.Sprintf("%.4f", v) },
	}).Parse(htmlReportTemplate)
	if err != nil {
		return err
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return tmpl.Execute(file, report)
}

func printSummary(report auditReport) {
	fmt.Println("\n[4/4] 审计完成")
	fmt.Println("-----------------------------------------------------")
	fmt.Printf("样本总数: %d\n", report.Metrics.Total)
	fmt.Printf("成员 / 非成员: %d / %d\n", report.Metrics.MemberSamples, report.Metrics.NonMembers)
	fmt.Printf("风险样本数: %d\n", report.Metrics.PredictedRisk)
	fmt.Printf("Accuracy: %.2f%%\n", report.Metrics.Accuracy*100)
	fmt.Printf("Precision: %.2f%%\n", report.Metrics.Precision*100)
	fmt.Printf("Recall: %.2f%%\n", report.Metrics.Recall*100)
	fmt.Printf("平均 shadow loss: %.4f\n", report.Metrics.MeanShadowLoss)
	fmt.Printf("平均边界距离: %.4f\n", report.Metrics.MeanDistance)
	fmt.Printf("平均波动系数: %.4f\n", report.Metrics.MeanVolatility)
	fmt.Println("-----------------------------------------------------")
}

func main() {
	cfg := loadConfigFromFlags()

	fmt.Println("=====================================================")
	fmt.Println("LabelScan-Go: Label-Only MIA 检测工具")
	fmt.Println("=====================================================")
	fmt.Printf("Preset: %s\nTarget: %s\nShadow: %s\n", cfg.Preset, cfg.TargetAPI, cfg.ShadowAPI)

	thresholds, err := loadThresholds(cfg.ShadowConfigPath)
	if err != nil {
		log.Fatalf("阈值配置加载失败: %v", err)
	}

	targetModel := client.NewHTTPClient(cfg.TargetAPI)
	shadowModel := client.NewHTTPClient(cfg.ShadowAPI)
	hsja := attack.NewHSJA(attack.HSJAConfig{
		MaxQueries:    cfg.MaxQueries,
		MaxIterations: cfg.MaxIterations,
		NumEvals:      cfg.NumEvals,
	})

	if err := calibrateThresholds(cfg, &thresholds, hsja, targetModel); err != nil {
		log.Fatalf("现场定标失败: %v", err)
	}
	fmt.Printf("定标阈值: TauD=%.4f TauCV=%.4f | Shadow red=%.4f yellow=%.4f\n",
		thresholds.TauD, thresholds.TauCV, thresholds.Tau95, thresholds.TauOpt)

	samples, err := loadAuditSamples(cfg)
	if err != nil {
		log.Fatalf("样本加载失败: %v", err)
	}

	fmt.Printf("\n[3/4] 执行 label-only 成员风险检测，workers=%d\n", cfg.AuditWorkers)
	engine := audit.NewEngine(thresholds, shadowModel, targetModel, hsja)
	pool := worker.NewAuditPool(engine, cfg.AuditWorkers)
	results := pool.RunAudit(samples)

	report := buildReport(cfg, thresholds, results)
	if err := writeJSONReport(cfg.JSONReportPath, report); err != nil {
		log.Fatalf("JSON 报告写入失败: %v", err)
	}
	if err := writeHTMLReport(cfg.HTMLReportPath, report); err != nil {
		log.Fatalf("HTML 报告写入失败: %v", err)
	}

	printSummary(report)
	fmt.Printf("JSON 报告: %s\n", cfg.JSONReportPath)
	if cfg.HTMLReportPath != "" {
		fmt.Printf("HTML 报告: %s\n", cfg.HTMLReportPath)
	}
}

const htmlReportTemplate = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LabelScan-Go Audit Report</title>
  <style>
    :root { color-scheme: light; --ink:#172026; --muted:#66727c; --line:#d9e1e7; --bg:#f6f8fa; --panel:#ffffff; --red:#b42318; --yellow:#9a6700; --orange:#c2410c; --green:#067647; --blue:#1f5eff; }
    body { margin:0; font-family:"Segoe UI", "Noto Sans SC", sans-serif; color:var(--ink); background:var(--bg); }
    header { padding:32px 40px 24px; background:#0f1720; color:#fff; }
    header h1 { margin:0 0 8px; font-size:30px; font-weight:700; }
    header p { margin:0; color:#b8c2cc; }
    main { max-width:1180px; margin:0 auto; padding:28px 24px 48px; }
    .grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin-bottom:22px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }
    .label { color:var(--muted); font-size:13px; margin-bottom:8px; }
    .value { font-size:26px; font-weight:750; }
    .section { background:var(--panel); border:1px solid var(--line); border-radius:8px; margin-top:18px; overflow:hidden; }
    .section h2 { font-size:18px; margin:0; padding:16px 18px; border-bottom:1px solid var(--line); }
    table { width:100%; border-collapse:collapse; font-size:14px; }
    th, td { padding:11px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { color:#42515c; background:#f9fafb; font-weight:650; }
    tr:last-child td { border-bottom:0; }
    .pill { display:inline-block; padding:3px 8px; border-radius:999px; font-weight:650; font-size:12px; }
    .risk-red { color:#fff; background:var(--red); }
    .risk-yellow { color:#402d00; background:#facc15; }
    .risk-orange { color:#fff; background:var(--orange); }
    .risk-green { color:#fff; background:var(--green); }
    .meta { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px 24px; padding:16px 18px; font-size:14px; }
    .meta span { color:var(--muted); }
    @media (max-width:860px) { .grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .meta { grid-template-columns:1fr; } }
    @media (max-width:560px) { header { padding:24px 20px; } main { padding:18px 12px 32px; } .grid { grid-template-columns:1fr; } table { font-size:12px; } th, td { padding:9px 8px; } }
  </style>
</head>
<body>
  <header>
    <h1>LabelScan-Go Audit Report</h1>
    <p>Generated at {{.GeneratedAt}} · preset {{.Config.Preset}}</p>
  </header>
  <main>
    <section class="grid">
      <div class="card"><div class="label">Accuracy</div><div class="value">{{pct .Metrics.Accuracy}}</div></div>
      <div class="card"><div class="label">Precision</div><div class="value">{{pct .Metrics.Precision}}</div></div>
      <div class="card"><div class="label">Recall</div><div class="value">{{pct .Metrics.Recall}}</div></div>
      <div class="card"><div class="label">Risk Rate</div><div class="value">{{pct .Metrics.HighRiskRate}}</div></div>
    </section>

    <section class="section">
      <h2>Run Summary</h2>
      <div class="meta">
        <div><span>Target API:</span> {{.Config.TargetAPI}}</div>
        <div><span>Shadow API:</span> {{.Config.ShadowAPI}}</div>
        <div><span>Samples:</span> {{.Metrics.Total}} total, {{.Metrics.MemberSamples}} member, {{.Metrics.NonMembers}} non-member</div>
        <div><span>Workers:</span> {{.Config.AuditWorkers}}</div>
        <div><span>Shadow thresholds:</span> red {{num .Thresholds.Tau95}}, yellow {{num .Thresholds.TauOpt}}</div>
        <div><span>Geometry thresholds:</span> distance {{num .Thresholds.TauD}}, volatility {{num .Thresholds.TauCV}}</div>
      </div>
    </section>

    <section class="section">
      <h2>Sample Results</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th><th>True Member</th><th>Risk</th><th>Label</th><th>Shadow Loss</th><th>Boundary Distance</th><th>Volatility</th>
          </tr>
        </thead>
        <tbody>
          {{range .Samples}}
          <tr>
            <td>{{.SampleID}}</td>
            <td>{{.IsMemberTrue}}</td>
            <td><span class="pill {{.RiskClass}}">{{.RiskLevel}}</span></td>
            <td>{{.Label}}</td>
            <td>{{num .ShadowLoss}}</td>
            <td>{{num .MeanDistance}}</td>
            <td>{{num .VolatilityCV}}</td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>`
