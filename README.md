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


### Examples

To run over QCD MC

```
> python ttbaranalysis.py --dataset QCD --iov 2016APV
```

Use `--test` to run over one file

```
> python ttbaranalysis.py --dataset QCD --iov 2016APV
```

Run using dask

```
> python ttbaranalysis.py --dataset QCD --iov 2016APV --dask
```