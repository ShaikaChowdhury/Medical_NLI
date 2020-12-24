# Medical_NLI

This repository contains the code for the paper,

Shaika Chowdhury, Philip S. Yu and Yuan Luo. Improving Medical NLI Using Context-Aware Domain Knowledge. Proceedings of the Ninth Joint Conference on Lexical and Computational Semantics (* SEM 2020). 2020.

## USAGE:

cd MUSKAN

cd scripts

cd training

python train_muskan.py

cd testing

python test_snli.py "mednli_test.pickle" ../../data/checkpoints/best.pth.tar


reference: https://github.com/coetaur0/ESIM
