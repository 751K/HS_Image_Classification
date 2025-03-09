import subprocess
import sys
import time


def run_main_with_input(input_value):
    try:
        process = subprocess.Popen([sys.executable, 'main.py'],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   bufsize=1)  # Line buffered

        # 发送输入
        process.stdin.write(f"{input_value}\n")
        process.stdin.flush()

        # 实时读取并打印输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

            # 同时检查错误输出
            error = process.stderr.readline()
            if error:
                print("错误:", error.strip(), file=sys.stderr)

        # 检查进程退出码
        if process.returncode != 0:
            print(f"进程以非零退出码结束: {process.returncode}")

    except Exception as e:
        import traceback
        error_msg = f"程序执行过程中发生错误:\n{str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"
        print(error_msg)


# 主执行逻辑
if __name__ == "__main__":
    for model_number in range(1, 16):  # 从1到12
        print(f"\n开始执行模型 {model_number}")
        run_main_with_input(model_number)
        print(f"模型 {model_number} 执行完成\n")
        time.sleep(2)
