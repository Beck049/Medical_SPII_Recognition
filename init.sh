#!/bin/bash

OS_TYPE="$(uname -s)"

# macOS 自動安裝 Homebrew
install_homebrew() {
    echo "開始安裝 Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [[ "$(uname -m)" == "arm64" ]]; then
        # M1/M2 Mac
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.bash_profile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        # Intel Mac
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.bash_profile
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    echo "Homebrew 安裝完成。"
}

# Windows 自動安裝 Chocolatey
install_chocolatey() {
    echo "開始安裝 Chocolatey..."
    powershell.exe -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command \
      "Set-ExecutionPolicy Bypass -Scope Process; \
       [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12; \
       iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    echo "Chocolatey 安裝完成。"
}

# 確保包管理工具存在
ensure_package_manager() {
    case "$OS_TYPE" in
        Darwin)
            if ! command -v brew &> /dev/null; then
                echo "Homebrew 未安裝，準備安裝..."
                install_homebrew
            else
                echo "Homebrew 已安裝。"
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*|Windows_NT)
            if ! command -v choco &> /dev/null; then
                echo "Chocolatey 未安裝，準備安裝..."
                install_chocolatey
            else
                echo "Chocolatey 已安裝。"
            fi
            ;;
        *)
            echo "不支援的系統類型：$OS_TYPE"
            exit 1
            ;;
    esac
}

# 安裝系統套件（brew/choco）
install_system_package() {
    local package_name=$1

    case "$OS_TYPE" in
        Darwin)
            echo "安裝系統套件 $package_name（使用 Homebrew）..."
            brew install $package_name
            ;;
        MINGW*|MSYS*|CYGWIN*|Windows_NT)
            echo "安裝系統套件 $package_name（使用 Chocolatey）..."
            choco install -y $package_name
            ;;
        *)
            echo "不支援的系統類型：$OS_TYPE"
            exit 1
            ;;
    esac
}

# 安裝 Python 套件
install_python_package() {
    local package_name=$1

    if ! command -v pip3 &> /dev/null; then
        echo "pip3 不存在，先安裝 pip3"
        if [[ "$OS_TYPE" == "Darwin" ]]; then
            brew install python3
        elif [[ "$OS_TYPE" == MINGW* || "$OS_TYPE" == MSYS* || "$OS_TYPE" == CYGWIN* || "$OS_TYPE" == Windows_NT ]]; then
            choco install -y python
        else
            echo "無法安裝 pip3，請手動安裝 Python 3"
            exit 1
        fi
    fi

    echo "更新 pip3..."
    pip3 install --upgrade pip

    if pip3 show $package_name &> /dev/null; then
        echo "Python 套件 $package_name 已安裝"
    else
        echo "安裝 Python 套件 $package_name ..."
        pip3 install $package_name
    fi
}

# 先確保包管理工具存在
ensure_package_manager

# 安裝 yq (系統套件)
if ! command -v yq &> /dev/null; then
    echo "沒有找到 yq，準備安裝..."
    install_system_package yq
else
    echo "yq 已安裝，版本：$(yq --version)"
fi

echo "開始從 config.yaml 讀取套件清單..."

# 用 yq 讀 config.yaml，分系統套件與 python 套件
system_packages=($(yq e '.system_packages[]' config.yaml))
python_packages=($(yq e '.python_packages[]' config.yaml))

# 安裝系統套件
for pkg in "${system_packages[@]}"; do
    if command -v $pkg &> /dev/null; then
        echo "系統套件 $pkg 已安裝"
    else
        install_system_package $pkg
    fi
done

# 安裝 Python 套件
for py_pkg in "${python_packages[@]}"; do
    install_python_package $py_pkg
done

echo "所有套件檢查與安裝完成！"
