package main

import (
	"fmt"
	"math"
	"testing"

	// ⚠️ 确保这个路径和你 go.mod 第一行的名字完全一致
	basic "label-only-mia-go/pkg/mathutils"
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