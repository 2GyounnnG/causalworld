# Paper-Ready Tables

## Table 1: rMD17 Aspirin Main H=16

| prior | mean +/- std | n |
| --- | ---: | ---: |
| none | 0.290 +/- 0.203 | 10 |
| Euclidean | 0.323 +/- 0.139 | 10 |
| spectral | 0.140 +/- 0.065 | 10 |

## Table 2: rMD17 Laplacian Ablation H=16

| Laplacian mode | mean +/- std | n |
| --- | ---: | ---: |
| per_frame | 0.111 +/- 0.075 | 5 |
| fixed_mean | 0.131 +/- 0.112 | 5 |
| fixed_frame0 | 0.146 +/- 0.089 | 5 |

## Table 3: rMD17 Weight Sweep Best H=16

| prior family | best weight | mean +/- std | n |
| --- | ---: | ---: | ---: |
| none | 0.001 | 0.417 +/- 0.243 | 3 |
| Euclidean | 0.01 | 0.173 +/- 0.060 | 3 |
| spectral | 0.1 | 0.130 +/- 0.098 | 3 |

## Table 4: Wolfram Flat H=16 Mean/Median

| prior | mean | median | std | n |
| --- | ---: | ---: | ---: | ---: |
| none | 0.074 | 0.064 | 0.046 | 10 |
| Euclidean | 0.065 | 0.050 | 0.041 | 10 |
| spectral | 18.3 | 0.130 | 45.4 | 10 |

## Table 5: Spectral Instability Seeds

| experiment | prior setting | seed | H=16 | H16/H1 |
| --- | --- | ---: | ---: | ---: |
| rMD17 aspirin weight sweep | spectral, w=0.01 | 1 | 3.274 | 68.6 |
| rMD17 aspirin weight sweep | spectral, w=0.01 | 2 | 2.339 | 51.8 |
| Wolfram flat 200ep | spectral | 1 | 3.486 | 946.3 |
| Wolfram flat 200ep | spectral | 2 | 143.6 | 4.9e4 |
| Wolfram flat 200ep | spectral | 8 | 35.2 | 8.8e3 |
