package main

import (
	"fmt"
	"log"
	
	// 引入刚才定义的 core 包
	// 注意：这里的路径是 "你的模块名/core"
	// 比如你们 go mod init mia-go，这里就是 "mia-go/core"
	"mia-go/core" 
	
	// 引入其他队员写的包
	// "mia-go/dataset" (队员 C)
	// "mia-go/model"   (队员 B)
	// "mia-go/attack"  (队长)
)

func main() {
	// 1. 队员 B 的工作：加载模型
	// 这里的 myModel 必须遵守 core.Model 接口
	// var myModel core.Model = &model.ONNXModel{} 
	// myModel.Load("target_model.onnx")

	// 2. 队员 C 的工作：加载数据
	// var myData []core.Sample
	// myData, _ = dataset.LoadCIFAR10("data/test_batch.bin")

	// 3. 队长的工作：初始化攻击者
	// var myAttacker core.Attacker = &attack.BoundaryAttack{}

	// 4. 开始干活
	// result := myAttacker.Attack(myModel, myData[0])
	
	// fmt.Printf("攻击结果: Distance=%f, Success=%v\n", result.Distance, result.IsSuccess)
}
