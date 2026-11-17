@echo off
setlocal
set PYTHONPATH=src
python tools/collect_metrics.py || exit /b 1
python tools/run_tests_with_coverage.py || exit /b 1
python tools/check_numbers.py --prove-failure || exit /b 1
echo ====================================================================
echo VERIFY SUCCEEDED: All metrics, coverage, and doc numbers green.
echo ====================================================================
