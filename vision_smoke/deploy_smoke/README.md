# ARM64 离线容器 smoke test

该目录生成一个 CPU-only、无第三方 Python 依赖、运行时完全离线的
`linux/arm64` Docker 镜像。容器读取 `/data/input.png`，执行固定的 gamma
校正，并写出 `/data/output.png`。

## 联网构建机

```bash
cd deploy_smoke
chmod +x build.sh
./build.sh
```

脚本优先使用 Docker buildx；没有 Docker daemon 时，自动通过 Docker Hub
Registry API 拉取官方 ARM64 Python 基础层并生成标准 Docker archive。
构建结果位于上一级目录：

```text
vision-runtime-arm64.tar
```

无 Docker 时仍可检查归档平台和内容：

```bash
python3 inspect_archive.py ../vision-runtime-arm64.tar
```

## 离线 ARM64 测试服务器

只需上传 `vision-runtime-arm64.tar` 和本目录中的 `input.png`，然后执行：

```bash
mkdir -p /root/vision_upload /root/vision_data
cd /root/vision_upload

docker load -i vision-runtime-arm64.tar
cp input.png /root/vision_data/input.png

docker run --rm \
  -v /root/vision_data:/data \
  vision-runtime:smoke

ls -lh /root/vision_data/input.png /root/vision_data/output.png
sha256sum /root/vision_data/input.png /root/vision_data/output.png
```

成功日志包含：

```text
arch=aarch64
python=3.10.x
input_size=256x256
output_size=256x256
status=container test passed
```

若输入文件缺失，容器会打印 `error=input file not found: /data/input.png` 并以
非零状态退出。

## 本机功能测试（不需要容器）

```bash
mkdir -p test_data
cp input.png test_data/input.png
python3 smoke.py --input test_data/input.png --output test_data/output.png
ls -lh test_data
```

`test_data/` 仅用于临时验证，不需要上传。
