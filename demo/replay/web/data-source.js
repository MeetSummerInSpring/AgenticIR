export class ReplayDataSource {
  constructor(manifestUrl = "data/current/manifest.json") {
    this.manifestUrl = manifestUrl;
  }

  async load() {
    const response = await fetch(this.manifestUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`无法读取回放数据（HTTP ${response.status}）`);
    }
    return response.json();
  }
}

// 实时版保持同一事件结构，通过 EventSource("/api/events") 逐条推送即可。
// 页面渲染逻辑不需要理解事件来自历史日志还是当前运行。
