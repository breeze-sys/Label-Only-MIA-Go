package core

// ==========================================
// 1. 全局配置与常量
// ==========================================
const (
	// 图片规格 (CIFAR-10)
	ImgChannels = 3
	ImgHeight   = 32
	ImgWidth    = 32

	// 展平后的向量长度: 3 * 32 * 32 = 3072
	FlattenedSize = ImgChannels * ImgHeight * ImgWidth
)

// ==========================================
// 2. 核心数据类型
// ==========================================

// Image 基础数据单元，使用 float32
type Image []float32

// Sample 代表一个测试样本
type Sample struct {
	ID       int    // 样本序号
	Data     Image  // 图片数据
	Label    int    // 真实标签 (Ground Truth)
	Filename string // 原文件名
}

// AttackResult 存储攻击结果 (用于写入 CSV)
type AttackResult struct {
	SampleID      int     // 样本 ID
	OriginalLabel int     // 原始标签
	FinalLabel    int     // 攻击后的标签
	IsSuccess     bool    // 攻击是否成功
	Queries       int     // 查询次数
	Distance      float64 // 最终 L2 距离 (MIA 核心指标)
	IsMember      bool    // 判定结果 (是否为训练集成员)
}

// ==========================================
// 3. 接口契约
// ==========================================

// Model 接口：队友 B (API Client) 必须实现此接口
// Attack 模块只通过这个接口与模型交互
type Model interface {
	// Predict 输入向量，返回 Label。
	// 错误处理：如果是网络错误，返回 error，Attack 应该处理重试或退出
	Predict(img Image) (int, error)
	PredictBatch(imgs []Image) ([]int,error)// 涡轮增压接口
	// GetInputSize 返回模型需要的输入维度 (3072)
	GetInputSize() int
}

// Attacker 接口：定义攻击逻辑的标准
type Attacker interface {
	// Attack 执行攻击
	// 输入: 原始样本, 目标模型
	// 输出: 攻击结果
	Attack(sample Sample, model Model) AttackResult
}
