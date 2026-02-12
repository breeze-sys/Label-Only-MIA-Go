package core

// ==========================================
// 1. 全局配置与常量 (Constants)
// ==========================================
const (
	// 图片规格 (CIFAR-10)
	ImgChannels = 3
	ImgHeight   = 32
	ImgWidth    = 32
	
	// 展平后的向量长度: 3 * 32 * 32 = 3072
	// 队员 C 在读取数据时，必须确保数据长度等于这个值
	FlattenedSize = ImgChannels * ImgHeight * ImgWidth

	// 攻击阈值 (示例值，具体需要在 main.go 中调整)
	// 如果 距离 > Threshold，判定为 Member
	DefaultThreshold = 0.5
)

// ==========================================
// 2. 核心数据类型 (Types)
// ==========================================

// Image 是最基础的数据单元
// 强制使用 float32，因为 ONNX Runtime 底层只认 float32
// 结构：一维数组，排列顺序必须是 NCHW 中的 CHW，即 [R所有, G所有, B所有]
type Image []float32

// Sample 代表一个测试样本
type Sample struct {
	ID       int    // 样本序号 (0-9999)
	Data     Image  // 图片数据
	Label    int    // 真实标签 (Ground Truth, 0-9)
	Filename string // 原文件名 (方便 Debug)
}

// AttackResult 存储攻击后的结果
// 最终会写入 CSV 文件用于 Python 画图
type AttackResult struct {
	SampleID      int     // 样本 ID
	OriginalLabel int     // 原始标签
	FinalLabel    int     // 攻击后的标签 (攻击成功时应与 Original 不同)
	IsSuccess     bool    // 攻击是否成功 (Label 是否改变)
	Queries       int     // 查询模型的次数 (Query Count)
	Distance      float64 // 最终距离 (L2 Norm)，这是 MIA 的核心指标
	IsMember      bool    // 你的预测：是否是训练集成语
}

// ==========================================
// 3. 接口契约 (Interfaces - 强制约束)
// ==========================================

// Model 接口：规定了队员 B 必须实现的方法
// 任何实现了这两个方法的结构体，都可以被当做 Model 使用
type Model interface {
	// Load 初始化模型 (加载 ONNX 文件)
	Load(path string) error

	// Predict 输入一张展平的图片，返回预测的类别 (0-9)
	// 为了性能，这里暂时不返回概率数组，只返回最终 Label
	// 如果需要概率，可以后续再加
	Predict(img Image) (int, error)
}

// Dataset 接口：规定了队员 C 必须实现的方法
type Dataset interface {
	// LoadBatch 加载指定数量的数据
	// path: 二进制文件的路径
	// limit: 加载多少张 (-1 代表全部)
	LoadBatch(path string, limit int) ([]Sample, error)
}

// Attacker 接口：规定了队长必须实现的逻辑
type Attacker interface {
	// Attack 对单个样本进行攻击
	// 输入: 模型接口, 样本数据
	// 输出: 攻击结果
	Attack(model Model, sample Sample) AttackResult
}
