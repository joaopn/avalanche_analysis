# avalanche_analysis

Simple avalanche analysis of timeseries. It depends on the following packages: `powerlaw, scipy`.
It accepts two types of data:

*a single timeseries `A = [1,2,0,1,1,0]` with separation of timescales (`A[i] = 0`).
*a 2-D array of separate trials: `A = [[1,2,1,0,0],[1,3,0,0,0]]`

Example usage:

```
import avalanche_analysis
A = avalanche_analysis.simulate_bp(m=1,trials=1000)
avalanche_analysis.run_analysis(data=A)
!["avalanche analysis"](example.png)
```