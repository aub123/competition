天池大赛平台支持选手以提交docker镜像的方式进行在线调试和运行自己的参赛代码，当前代码包是本地打包镜像的一个demo，
实际使用时您可以使用其他符合自己需求的linux基础镜像进行打包，安装自己需要的基础环境或扩展包，镜像打包
需符合以下要求：
- 需安装curl/zip/unzip等基础命令
- 按照赛题说明文档约定将代码和启动命令复制到指定位置
- 按照赛题说明文档约定将结果文件写入到指定位置

参赛步骤：
1.根据赛题要求编写代码
2.将代码及需要的安装包打包成镜像
3.将镜像推送到镜像仓库中(推荐推送到阿里云镜像仓库，地域根据赛题说明约定)
4.大赛详情页提交镜像地址、用户名和密码


资料附录：
1.使用pip安装python扩展时，可以使用国内的源以加快下载速度
  1)阿里云
  使用参考 https://developer.aliyun.com/mirror/pypi
  安装示例 pip3 install numpy --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

  2)清华
  使用参考：https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
  安装示例 pip install numpy --index-url=https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host=pypi.tuna.tsinghua.edu.cn

2.基础镜像及安装源配置参考
  ubuntu系统
  阿里云: https://developer.aliyun.com/mirror/ubuntu
  清华：https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/

3.docker镜像介绍课程
  https://tianchi.aliyun.com/course/351






