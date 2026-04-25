package main

import (
	"fmt"
	"math"
	"testing"

	// ⚠️ 确保这个路径和你 go.mod 第一行的名字完全一致
	basic "Label-Only-MIA-Go/pkg/mathutils"
)

// 辅助函数：用于比较两个浮点数切片是否相等（允许微小误差）
func vectorsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	const epsilon = 1e-5 // 允许的浮点误差
	for i := range a {
		// 注意：这里需要 float64 转换，因为 math.Abs 接收 float64
		if math.Abs(float64(a[i]-b[i])) > epsilon {
			return false
		}
	}
	return true
}

// 辅助函数：格式化打印向量
func printVec(name string, v []float32) {
	fmt.Printf("  %s: %v\n", name, v)
}

func TestNewVector(t *testing.T) {
	fmt.Println("=== 测试 NewVector ===")
	// ⚠️ 修正：添加 basic. 前缀
	got := basic.NewVector(3, 2.5)
	want := []float32{2.5, 2.5, 2.5}

	printVec("结果", got)

	if !vectorsEqual(got, want) {
		t.Errorf("NewVector 失败: 期望 %v, 实际 %v", want, got)
	}
}

func TestVectorAdd(t *testing.T) {
	fmt.Println("=== 测试 VectorAdd ===")
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{0.5, 0.5, 0.5}

	// ⚠️ 修正：添加 basic. 前缀
	got := basic.VectorAdd(a, b)
	want := []float32{1.5, 2.5, 3.5}

	printVec("A+B", got)
	if !vectorsEqual(got, want) {
		t.Errorf("VectorAdd 失败")
	}
}

func TestVectorSub(t *testing.T) {
	fmt.Println("=== 测试 VectorSub ===")
	a := []float32{10.0, 20.0, 30.0}
	b := []float32{1.0, 2.0, 3.0}

	// ⚠️ 修正：添加 basic. 前缀
	got := basic.VectorSub(a, b)
	want := []float32{9.0, 18.0, 27.0}

	printVec("A-B", got)
	if !vectorsEqual(got, want) {
		t.Errorf("VectorSub 失败")
	}
}

func TestVectorMul(t *testing.T) {
	fmt.Println("=== 测试 VectorMul ===")
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{2.0, 0.5, -1.0}

	// ⚠️ 修正：添加 basic. 前缀
	got := basic.VectorMul(a, b)
	want := []float32{2.0, 1.0, -3.0}

	printVec("A*B", got)
	if !vectorsEqual(got, want) {
		t.Errorf("VectorMul 失败")
	}
}

func TestVectorScale(t *testing.T) {
	fmt.Println("=== 测试 VectorScale ===")
	v := []float32{1.0, -2.0, 0.5}

	// ⚠️ 修正：添加 basic. 前缀
	got := basic.VectorScale(v, 2.0)
	want := []float32{2.0, -4.0, 1.0}

	printVec("Scale", got)
	if !vectorsEqual(got, want) {
		t.Errorf("VectorScale 失败")
	}
}

func TestClip(t *testing.T) {
	fmt.Println("=== 测试 Clip ===")
	v := []float32{-1.5, 0.5, 1.5}

	// ⚠️ 修正：添加 basic. 前缀
	got := basic.Clip(v, 0.0, 1.0)
	want := []float32{0.0, 0.5, 1.0}

	printVec("Clip", got)
	if !vectorsEqual(got, want) {
		t.Errorf("Clip 失败")
	}
}

func TestClone(t *testing.T) {
	fmt.Println("=== 测试 Clone ===")
	origin := []float32{1.0, 2.0}

	// ⚠️ 修正：添加 basic. 前缀
	cloned := basic.Clone(origin)

	cloned[0] = 999.0
	if origin[0] == 999.0 {
		t.Errorf("Clone 是浅拷贝！")
	} else {
		fmt.Println("  深拷贝验证通过")
	}
}

func TestMeanAndStd(t *testing.T) {
	fmt.Println("=== 测试 MeanAndStd ===")
	// 测试数据：1, 2, 3, 4, 5
	// 期望平均值：3.0
	// 样本方差 (n-1)：10 / 4 = 2.5
	// 样本标准差：sqrt(2.5) ≈ 1.5811388
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	mean, std := basic.MeanAndStd(data)
	fmt.Printf("  Mean: %v, Std: %v\n", mean, std)

	if math.Abs(mean-3.0) > 1e-5 {
		t.Errorf("Mean 计算错误: 期望 3.0, 实际 %v", mean)
	}
	if math.Abs(std-1.5811388) > 1e-5 {
		t.Errorf("Std 计算错误: 期望 ~1.5811388, 实际 %v", std)
	}
}

func TestCalibrateReference(t *testing.T) {
	fmt.Println("=== 测试 CalibrateReference ===")
	// 模拟3个路人的数据，每个路人测试3张图，仅用于跑通逻辑
	// 注意真实场景是 10 个路人
	dists := [][]float64{
		{1.0, 1.1, 0.9}, // Mean: 1.0, Std: 0.1, CV: 0.1
		{2.0, 2.2, 1.8}, // Mean: 2.0, Std: 0.2, CV: 0.1
		{3.0, 3.3, 2.7}, // Mean: 3.0, Std: 0.3, CV: 0.1
	}

	tauD, tauCV := basic.CalibrateReference(dists)
	fmt.Printf("  tauD: %v, tauCV: %v\n", tauD, tauCV)

	// dBarList: {1.0, 2.0, 3.0} => muD: 2.0, sD: 1.0 => tauD: 2.0 + 1.2 * 1.0 = 3.2
	// cvList: {0.1, 0.1, 0.1} => muCV: 0.1, sCV: 0.0 => tauCV: 0.1 - 1.92 * 0.0 = 0.1
	if math.Abs(tauD-3.2) > 1e-5 {
		t.Errorf("tauD 计算错误: 期望 3.2, 实际 %v", tauD)
	}
	if math.Abs(tauCV-0.1) > 1e-5 {
		t.Errorf("tauCV 计算错误: 期望 0.1, 实际 %v", tauCV)
	}
}
