# avalanche_analysis

Simple avalanche analysis of timeseries. It depends on the following packages: `powerlaw, scipy`.
It accepts two types of data:

*a single timeseries `A = [1,2,0,1,1,0]` with separation of timescales (`A[i] = 0`).
*a list of discrete avalanches `A = [[1,2],[1,1]]`.

To run it we use:

```
import avalanche_analysis
avalanche_analysis.run_analysis(data=A)
!["avalanche analysis"](analysis.png)
```