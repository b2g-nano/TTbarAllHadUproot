# ttbarallhadronic

## Getting started

### on the LPC

```
ssh -L localhost:8xxx:localhost:8xxx  <username>@cmslpc-sl7.fnal.gov
```

```
git clone ssh://git@gitlab.cern.ch:7999/mmorris/ttbarallhadronic.git
cd ttbarallhadronic
bash bootstrap.sh
./shell
> jupyter notebook --ip 0.0.0.0 --no-browser --port=8xxx
```


### Running the background estimation

1. Run the processor to save unweighted JetHT and TTbar MC files needed for the mistag rate calculation

```
> python ttbaranalysis.py --dataset JetHT --iov 2016APV
> python ttbaranalysis.py --dataset TTbar --iov 2016APV

> python ttbaranalysis.py --dataset JetHT --iov 2016
> python ttbaranalysis.py --dataset TTbar --iov 2016
```

2. Run the processor to save unweighted QCD MC, needed for the mass modification procedure

```
> python ttbaranalysis.py --dataset QCD --iov 2016APV
> python ttbaranalysis.py --dataset QCD --iov 2016

```

3. Calculate the mistag rate and save csv files

    Run `mistag/mistagRate.ipynb` notebook
    </br>
4. After running over JetHT, TTbar, and QCD as above, run over JetHT with the mistag rate and QCD mass modfication applied

```
> python ttbaranalysis.py --dataset JetHT --iov 2016APV --bkgest
> python ttbaranalysis.py --dataset JetHT --iov 2016 --bkgest
```




### other examples


Use `--test` to run over one file

```
> python ttbaranalysis.py --dataset QCD --iov 2016APV
```

Run using dask

```
> python ttbaranalysis.py --dataset QCD --iov 2016APV --dask
```