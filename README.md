## Getting started

### LPC

```
ssh -L localhost:8xxx:localhost:8xxx  <username>@cmslpc-sl7.fnal.gov
```

```
git clone https://github.com/b2g-nano/TTbarAllHadUproot.git
cd TTbarAllHadUproot
git checkout optimize
bash bootstrap.sh
./shell
> jupyter notebook --ip 0.0.0.0 --no-browser --port=8xxx
```

### Coffea Casa

Log in to https://coffea.casa with CERN SSO

***dask not set up for Coffea Casa***


```
git clone https://github.com/b2g-nano/TTbarAllHadUproot.git
cd TTbarAllHadUproot
git checkout optimize
```

To run a test on QCD, TTbar, and JetHT for each year:

```
python ttbaranalysis.py --iov 2016APV --test --env casa
python ttbaranalysis.py --iov 2016 --test --env casa
python ttbaranalysis.py --iov 2017 --test --env casa
python ttbaranalysis.py --iov 2018 --test --env casa
```


### Running the transfer function background estimation

1. Run the processor to save coffea files needed for the background estimation

```
python ttbaranalysis.py --iov 2016 --dask

```

2. Convert the coffea histograms to root histograms needed for 2Dalpabet in `scaleCoffeaFiles.ipynb`

3. Copy the root files over to lxplus and follow the instructions here: https://github.com/mdmorris/BstarToTW_CMSDAS2023_BackgroundEstimation


### Running the mistag rate background estimation

1. Run the processor to save coffea files needed for the background estimation

```
> python ttbaranalysis.py --noSyst --iov 2016 --dask
```

2. Calculate the mistag rate and save in a csv file in `data/corrections/backgroundEstimate`

```
python python/mistag.py

```

3. Run the processor with the background estimation. Files saved in `outputs`

```
python ttbaranalysis.py --dataset JetHT --iov 2016 --dask --bkgest mistag
```



### Run in parallel for faster processing (LPC only)

To run in parallel, ssh into a separate LPC node for each sample you wish to run. For example if you wish to run 6 samples, in separate terminals:

```
ssh -XY <user>@cmslpc110.fnal.gov
ssh -XY <user>@cmslpc111.fnal.gov
ssh -XY <user>@cmslpc112.fnal.gov
ssh -XY <user>@cmslpc113.fnal.gov
ssh -XY <user>@cmslpc114.fnal.gov
ssh -XY <user>@cmslpc115.fnal.gov
```

To run the background samples for 2016, in 6 separate LPC nodes run:

```
python ttbaranalysis.py --iov 2016 --dataset JetHT --era F --dask
```
```
python ttbaranalysis.py --iov 2016 --dataset JetHT --era G --dask
```
```
python ttbaranalysis.py --iov 2016 --dataset JetHT --era H --dask
```
```
python ttbaranalysis.py --iov 2016 --dataset TTbar --pt 700to1000 --dask
```
```
python ttbaranalysis.py --iov 2016 --dataset TTbar --pt 1000toInf --dask
```
```
python ttbaranalysis.py --iov 2016 --dataset QCD --dask
```

### Miscellaneous

- Use `--test` to run over one file
- plotting performed in `plots/plotting.py`
- plot interactively with `plots/viewPlots.ipynb`

