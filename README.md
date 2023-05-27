## Getting started

### LPC

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

### Coffea Casa

Log in to https://coffea.casa with CERN SSO

```
git clone ssh://git@gitlab.cern.ch:7999/mmorris/ttbarallhadronic.git
cd ttbarallhadronic
```

### Running the background estimation

1. Run the processor to save coffea files needed for the background estimation
   - JetHT, TTbar saved in `outputs/` to calculate the mistag rate
   - QCD saved in `data/corrections/backgroundEstimation` for the mass modification procedure

```
> python ttbaranalysis.py --dataset JetHT --iov 2016 --dask
> python ttbaranalysis.py --dataset TTbar --iov 2016 --dask
> python ttbaranalysis.py --dataset QCD --iov 2016 --dask
```


2. Calculate the mistag rate and save in a csv file in `data/corrections/backgroundEstimation`

```
> python python/mistag.py

```

3. Run the processor with the background estimation. Files saved in `outputs`

```
> python ttbaranalysis.py --dataset JetHT --iov 2016 --bkgest --dask
> python ttbaranalysis.py --dataset TTbar --iov 2016 --bkgest --dask
> python ttbaranalysis.py --dataset QCD --iov 2016 --bkgest --dask
```



### other examples

- Use `--test` to run over one file

- plot the closureTest with plots/closureTest.ipynb

- plot the mistagRate with mistag/mistagRate.ipynb

