package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"Label-Only-MIA-Go/pkg/core"
)

// ==========================================
// 1. 结构体与通信协议定义
// ==========================================

type HTTPClient struct {
	url        string // 基础地址，如 http://localhost:8000
	httpClient *http.Client
}

// 单图请求体
type requestBody struct {
	Image []float32 `json:"image"`
}

// 标签返回体 (用于 Predict)
type responseBody struct {
	Label int `json:"label"`
}

// 分数返回体 (用于 PredictLogits - 核心新增！)
type logitsResponse struct {
	Logits []float32 `json:"logits"` // 对应 Python 返回的 [0.1, 0.8, ...]
}

// 批量请求体 (用于 PredictBatch)
type batchRequest struct {
	Images [][]float32 `json:"images"`
}

type batchResponse struct {
	Labels []int `json:"labels"`
}

// ==========================================
// 2. 初始化：NewHTTPClient (对齐 main.go 的调用)
// ==========================================

func NewHTTPClient(baseURL string) *HTTPClient {
	t := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     90 * time.Second,
	}

	return &HTTPClient{
		url: baseURL,
		httpClient: &http.Client{
			Transport: t,
			Timeout:   15 * time.Second, // 边界攻击计算慢，建议给 15s 超时
		},
	}
}

// ==========================================
// 3. 核心功能实现 (实现 core.Model 接口)
// ==========================================

// Predict: 获取最终标签
func (c *HTTPClient) Predict(img core.Image) (int, error) {
	// 注意：根据 A 的接口规范，单图预测通常在 /predict 路径
	endpoint := c.url + "/predict"

	payload := requestBody{Image: img}
	jsonData, _ := json.Marshal(payload)

	resp, err := c.httpClient.Post(endpoint, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return -1, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return -1, fmt.Errorf("predict 接口报错: %d", resp.StatusCode)
	}

	var result responseBody
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return -1, err
	}
	return result.Label, nil
}

// PredictLogits: 获取原始分数 (用于算方案一的 Loss) —— 【这是你最需要的改动】
func (c *HTTPClient) PredictLogits(img core.Image) ([]float32, error) {
	// 注意：路径设为 /predict_logits
	endpoint := c.url + "/predict_logits"

	payload := requestBody{Image: img}
	jsonData, _ := json.Marshal(payload)

	resp, err := c.httpClient.Post(endpoint, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Logits接口报错: %d", resp.StatusCode)
	}

	var result logitsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Logits, nil
}

// PredictBatch: 批量预测标签 (用于 HSJA 涡轮增压)
func (c *HTTPClient) PredictBatch(imgs []core.Image) ([]int, error) {
	if len(imgs) == 0 {
		return []int{}, nil
	}

	rawImgs := make([][]float32, len(imgs))
	for i, img := range imgs {
		rawImgs[i] = img
	}

	payload := batchRequest{Images: rawImgs}
	jsonData, _ := json.Marshal(payload)

	// 路径设为 /predict_batch
	batchURL := c.url + "/predict_batch"

	resp, err := c.httpClient.Post(batchURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("predict_batch 接口报错: %d", resp.StatusCode)
	}

	var result batchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.Labels, nil
}

func (c *HTTPClient) GetInputSize() int {
	return 3072
}
