// 页面导航功能
function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(sectionId).classList.add('active');

    // 更新导航栏活动状态
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });

    if(sectionId === 'generate') {
        document.querySelector('.nav-link[href="#"][onclick="showSection(\'generate\')"]').classList.add('active');
    } else if(sectionId === 'profile') {
        document.querySelector('.nav-link[href="#"][onclick="showSection(\'profile\')"]').classList.add('active');
    } else if(sectionId === 'login') {
        document.querySelector('.nav-link[href="#"][onclick="showSection(\'login\')"]').classList.add('active');
    } else {
        document.querySelector('.nav-link[href="#"][onclick="showSection(\'home\')"]').classList.add('active');
    }

    window.scrollTo(0, 0);
}

// 文件拖拽上传功能
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const generateBtn = document.getElementById('generate-btn');
const loading = document.getElementById('loading');
const resultPlaceholder = document.getElementById('result-placeholder');
const resultContent = document.getElementById('result-content');
const resultImage = document.getElementById('result-image');
const memeText = document.getElementById('meme-text');

// 防止默认拖拽行为
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// 高亮拖拽区域
['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
});

function highlight() {
    dropArea.style.backgroundColor = 'rgba(67, 97, 238, 0.15)';
    dropArea.style.borderColor = '#3a0ca3';
}

function unhighlight() {
    dropArea.style.backgroundColor = 'rgba(67, 97, 238, 0.05)';
    dropArea.style.borderColor = '#4361ee';
}

// 处理文件放置
dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

// 处理文件选择
fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        const fileType = file.type;

        if (fileType.match('image.*')) {
            const reader = new FileReader();

            reader.onload = function(e) {
                // 显示上传的图片预览
                const imgPreview = document.createElement('img');
                imgPreview.src = e.target.result;
                imgPreview.style.maxWidth = '100%';
                imgPreview.style.maxHeight = '250px';
                imgPreview.style.borderRadius = '8px';

                dropArea.innerHTML = '';
                dropArea.appendChild(imgPreview);

                // 启用生成按钮
                generateBtn.disabled = false;
            };

            reader.readAsDataURL(file);
        } else {
            alert('请上传图片文件 (JPG, PNG)');
        }
    }
}

// 生成表情包（使用AJAX）
document.getElementById('generate-btn').addEventListener('click', async function() {
    const fileInput = document.getElementById('file-input');

    if (fileInput.files.length === 0) {
        alert('请先选择图片');
        return;
    }

    // 显示加载动画
    loading.style.display = 'block';
    resultPlaceholder.style.display = 'none';
    resultContent.style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        const response = await fetch('/generate_meme', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.status !== 200) {
            throw new Error(result.error || '生成失败');
        }

        // 更新结果展示
        resultImage.src = result.meme;
        // document.querySelector('.emotion-tag').textContent = `表情: ${emotion}`;
        resultContent.style.display = 'block';

    } catch (error) {
        console.error('Error:', error);
        alert(`错误: ${error.message}`);
    } finally {
        // 隐藏加载动画
        loading.style.display = 'none';
    }
});

// 初始化页面
document.addEventListener('DOMContentLoaded', function() {
    // 禁用生成按钮，直到上传图片
    generateBtn.disabled = true;
});