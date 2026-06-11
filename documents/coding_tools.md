# 1 反向端口转发
使用ikun美国节点，打开tun模式
反向端口转发：
ssh -R 7890:localhost:7890 -i "C:\Users\zyc\Desktop\sshkey" root@121.89.91.224

# 2 设置服务器的ip
vim ~/.bashrc

加在末尾
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

source ~/.bashrc

验证
curl ip-api.com

# 下载cli和chat版本
下载npm
apt update
apt install -y curl
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt install -y nodejs

下载cli版本
sudo npm install -g @openai/codex
sudo npm install -g @anthropic-ai/claude-code

下载chat版本 ...

# 设置coding agent的ip
codex不用换直接登
claude先在终端中登录
然后打开 ~/.claude/settings.json覆盖为
{
  "env": {
    "HTTP_PROXY": "http://127.0.0.1:7890",
    "HTTPS_PROXY": "http://127.0.0.1:7890",
    "ALL_PROXY": "socks5://127.0.0.1:7890"
  },
  "theme": "dark"
}



