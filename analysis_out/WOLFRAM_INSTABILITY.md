# Wolfram Instability

Instability is flagged when `H16/H1 > 20` or `H16 > 1.0`.

## Aggregate H16/H1

| prior | h1 | h16 | h16_over_h1 | unstable | reasons |
| --- | --- | --- | --- | --- | --- |
| none | 0.003376554593 | 0.07396286912 | 21.90483438 | true | H16/H1 > 20 |
| euclidean | 0.007710161293 | 0.06538378596 | 8.480209878 | false |  |
| spectral | 0.006562171201 | 18.29436963 | 2787.853146 | true | H16/H1 > 20; H16 > 1 |

## Seed-Level Instability Flags

| prior | seed | h1 | h16 | h16_over_h1 | reasons |
| --- | --- | --- | --- | --- | --- |
| euclidean | 8 | 0.003524931148 | 0.1016481146 | 28.83690783 | H16/H1 > 20 |
| euclidean | 9 | 0.006807972677 | 0.1440101415 | 21.15316091 | H16/H1 > 20 |
| none | 0 | 0.001808356494 | 0.04923725128 | 27.22762434 | H16/H1 > 20 |
| none | 1 | 0.006368814968 | 0.1702914685 | 26.73832877 | H16/H1 > 20 |
| none | 3 | 0.003249018686 | 0.09071860462 | 27.92184761 | H16/H1 > 20 |
| none | 4 | 0.001155871549 | 0.02387126163 | 20.65217511 | H16/H1 > 20 |
| none | 5 | 0.00542414654 | 0.1257547736 | 23.18425077 | H16/H1 > 20 |
| none | 7 | 0.002802271629 | 0.0779568553 | 27.81916446 | H16/H1 > 20 |
| none | 8 | 0.002128578722 | 0.04289799929 | 20.15335342 | H16/H1 > 20 |
| spectral | 0 | 0.002999092219 | 0.06530684978 | 21.77553906 | H16/H1 > 20 |
| spectral | 1 | 0.00368359033 | 3.485707521 | 946.2799087 | H16/H1 > 20; H16 > 1 |
| spectral | 2 | 0.002933676355 | 143.5640717 | 48936.57456 | H16/H1 > 20; H16 > 1 |
| spectral | 3 | 0.003259699559 | 0.1593713611 | 48.89142642 | H16/H1 > 20 |
| spectral | 4 | 0.002485054545 | 0.1124669686 | 45.25734408 | H16/H1 > 20 |
| spectral | 8 | 0.003992063925 | 35.24406433 | 8828.532057 | H16/H1 > 20; H16 > 1 |

## Spectral Seed-Level H=16

| seed | h1 | h16 | h16_over_h1 | unstable | reasons |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.002933676355 | 143.5640717 | 48936.57456 | true | H16/H1 > 20; H16 > 1 |
| 8 | 0.003992063925 | 35.24406433 | 8828.532057 | true | H16/H1 > 20; H16 > 1 |
| 1 | 0.00368359033 | 3.485707521 | 946.2799087 | true | H16/H1 > 20; H16 > 1 |
| 3 | 0.003259699559 | 0.1593713611 | 48.89142642 | true | H16/H1 > 20 |
| 6 | 0.01720698737 | 0.148344934 | 8.621203168 | false |  |
| 4 | 0.002485054545 | 0.1124669686 | 45.25734408 | true | H16/H1 > 20 |
| 7 | 0.0198870264 | 0.06917738169 | 3.478518121 | false |  |
| 0 | 0.002999092219 | 0.06530684978 | 21.77553906 | true | H16/H1 > 20 |
| 5 | 0.003740634071 | 0.06162333488 | 16.47403454 | false |  |
| 9 | 0.005433887243 | 0.03356193379 | 6.176413364 | false |  |

## Mean Vs Median H=16

| prior | h16_mean | h16_median | h16_std | n |
| --- | --- | --- | --- | --- |
| none | 0.07396286912 | 0.06359705329 | 0.04574169559 | 10 |
| euclidean | 0.06538378596 | 0.05016138218 | 0.0411988889 | 10 |
| spectral | 18.29436963 | 0.1304059513 | 45.36049594 | 10 |

## Best H=16 Setting

| setting | h16 | n |
| --- | --- | --- |
| none mean | 0.07396286912 | 10 |
| euclidean mean | 0.06538378596 | 10 |
| spectral mean | 18.29436963 | 10 |
| spectral median | 0.1304059513 | 10 |

Best H=16 mean setting: `euclidean`.

## Interpretation

Euclidean has the best H16 mean in the completed Wolfram flat 200ep run.

Spectral has heavy-tailed long-horizon instability.

Spectral seed 1, seed 2, and seed 8 explode at H=16.

Most spectral seeds remain near baseline scale, so the failure is heavy-tailed rather than uniform.

Wolfram spectral has heavy-tailed long-horizon instability; 7/10 seeds are near baseline scale, but seeds 1, 2, and 8 explode at H=16.

## Notes

- This script reads existing analysis artifacts only.
- Training code and raw result JSONs are not modified.
