nohup python scripts/run_pipeline_ubuntu.py --target sn --n-trials 1000 --n-jobs 3 --loco-max-groups 0 --run-name run_sn_basic > log_sn_basic.txt 2>&1 &

nohup python scripts/run_pipeline_ubuntu.py --target sn --use-extended --n-trials 1000 --n-jobs 3 --loco-max-groups 0 --run-name run_sn_ext > log_sn_ext.txt 2>&1 &

nohup python scripts/run_pipeline_ubuntu.py --target s2n --n-trials 1000 --n-jobs 3 --loco-max-groups 0 --run-name run_s2n_basic > log_s2n_basic.txt 2>&1 &

nohup python scripts/run_pipeline_ubuntu.py --target s2n --use-extended --n-trials 1000 --n-jobs 3 --loco-max-groups 0 --run-name run_s2n_ext > log_s2n_ext.txt 2>&1 &