package attack

// TargetModel 定义了黑盒模型必须具备的能力
type TargetModel interface {
	// Predict 接收一维化的图片数据 ([]float32)，返回预测的 Label (int)
	// 对应 Python 中的 model.predict(x) -> label
	Predict(input []float32) int

	// InputSize 返回模型需要的输入维度 (例如 3072)
	InputSize() int
}
