## Setup

On UMIACS server

- Checkout: git clone git@github.com:vietansegan/herbal.git
- Set $SEGAN to /fs/clip-political/vietan/segan
- Set $GTP to /fs/clip-political/vietan/gtpounder
- Build: ant jar

## Run

### To learn HIPTM on GOP data

```
 java -Xmx5000M -Xms5000M -cp "dist/herbal.jar:lib/*:$SEGAN/dist/lib/*:$GTP/dist/*" experiment.percongress.GOPExpt --congress-num 112 -v -d --processed-data-folder /fs/clip-political/vietan/herbal-clip-ml/herbal/data/govtrack/112/house-5K --expt-folder experiments/house-5K-topichier --run-mode run --model hier-mult-shdp --alpha 0.1 --beta 0.1 --rho 0.25 --sigma 25.0 --mu 0.0 --gamma 25.0 --epsilon 1.0 --burnIn 500 --maxIter 1000  --sampleLag 100 --report 50 --local-alpha 5 --global-alpha 10 --init-maxiter 1000 -analyze -train
```

### To generate the HTML file for the learned model

```
 java -Xmx5000M -Xms5000M -cp "dist/herbal.jar:lib/*:$SEGAN/dist/lib/*:$GTP/dist/*" experiment.percongress.GOPExpt --congress-num 112 -v -d --processed-data-folder /fs/clip-political/vietan/herbal-clip-ml/herbal/data/govtrack/112/house-5K --expt-folder experiments/house-5K-topichier --run-mode run --model hier-mult-shdp --alpha 0.1 --beta 0.1 --rho 0.25 --sigma 25.0 --mu 0.0 --gamma 25.0 --epsilon 1.0 --burnIn 500 --maxIter 1000  --sampleLag 100 --report 50 --local-alpha 5 --global-alpha 10 --init-maxiter 1000 -analyze -html
```
