@echo off
setlocal

REM 创建logs目录（如果不存在）
if not exist logs mkdir logs

REM 获取当前日期时间作为日志文件名的一部分
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
set LOG_FILE=logs\api_%TIMESTAMP%.log

echo Starting BERT Classification API on http://0.0.0.0:6565
echo Logs will be saved to %LOG_FILE%

REM 启动服务并将输出重定向到日志文件
uvicorn api.main:app --host 0.0.0.0 --port 6565 --log-level info > %LOG_FILE% 2>&1
