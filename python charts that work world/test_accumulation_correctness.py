"""Unit tests for corrections to WRF accumulation scripts; no WRF installation needed."""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wrf_accumulation_fix import accumulated_difference, exact_time_pairs, safe_accumulation, assert_same_geography

class Var:
 def __init__(self,data, time=True):
  self.arr=np.asarray(data,dtype=float)
  self.dimensions=('Time','south_north','west_east') if time else ('south_north','west_east')
 def __getitem__(self,key):return self.arr[key]
class Dataset:
 def __init__(self, value=0,lat=30.0,lon=-95.0, add_rainsh=False, add_snow=True, missing=False):
  xy=np.zeros((2,3))+value
  self.variables={'XLAT':Var(np.full((1,2,3),lat)),'XLONG':Var(np.full((1,2,3),lon)),
       'RAINC':Var(xy[None]*0.25),'RAINNC':Var(xy[None]*0.75)}
  if add_rainsh:self.variables['RAINSH']=Var(xy[None]*0.1)
  if add_snow:self.variables['ACSNOW']=Var(xy[None])
  if missing:self.variables.pop('RAINNC')

def daterange(minutes=15,hours=25):
 base=datetime(2026,6,1)
 return [(f'f{i}',0,base+timedelta(minutes=i*minutes)) for i in range((hours*60)//minutes+1)]

def test_15_minute_one_hour():
 rows=exact_time_pairs(daterange(),1)
 assert rows[0]==('f4',0,'f0',0)
 assert len(rows)==97

def test_15_minute_three_hour():
 rows=exact_time_pairs(daterange(),3)
 assert rows[0]==('f12',0,'f0',0)

def test_15_minute_24hour():
 rows=exact_time_pairs(daterange(),24)
 assert rows[0]==('f96',0,'f0',0)

def test_3_hourly_one_hour_unavailable():
 assert exact_time_pairs(daterange(minutes=180,hours=6),1)==[]

def test_missing_timestamp_skipped():
 seq=daterange(minutes=15,hours=4)
 seq=[r for r in seq if r[2]!=datetime(2026,6,1,0,15)]
 assert ('f5',0,'f1',0) not in exact_time_pairs(seq,1)

def test_tzaware_normalizes_to_utc():
 a=datetime(2026,6,1,0,0,tzinfo=timezone.utc)
 b=datetime(2026,6,1,2,0,tzinfo=timezone(timedelta(hours=1)))
 assert exact_time_pairs([('a',0,a),('b',0,b)],1)==[('b',0,'a',0)]

def test_duplicate_timestamp_raises():
 t=datetime(2026,6,1)
 with pytest.raises(ValueError,match='Duplicate'):
  exact_time_pairs([('a',0,t),('b',1,t)],1)

def test_precip_mm_and_inch():
 start,end=Dataset(10),Dataset(15)
 delta=accumulated_difference(end,0,start,0,'rain')
 np.testing.assert_allclose(delta,5)
 np.testing.assert_allclose(delta/25.4,0.19685039370078738)

def test_optional_rainsh_both_ends():
 assert np.allclose(accumulated_difference(Dataset(15,add_rainsh=True),0,Dataset(10,add_rainsh=True),0,'rain'),5.5)

def test_optional_rainsh_mismatch_rejected():
 with pytest.raises(ValueError,match='RAINSH'):
  accumulated_difference(Dataset(15,add_rainsh=True),0,Dataset(10),0,'rain')

def test_snow_water_equivalent_units():
 q=accumulated_difference(Dataset(15),0,Dataset(10),0,'snow_swe')
 np.testing.assert_allclose(q,5)
 assert np.isclose((q/25.4)[0,0],0.19685039370078738)
 assert np.isclose((q*10)[0,0],50) # Explicit 10:1 depth *estimate*, not SWE

def test_missing_accumulator_refuses_guess():
 with pytest.raises(KeyError,match='ACSNOW'):
  accumulated_difference(Dataset(15,add_snow=False),0,Dataset(10),0,'snow_swe')

def test_missing_precip_refuses_guess():
 with pytest.raises(KeyError,match='RAINNC'):
  accumulated_difference(Dataset(15,missing=True),0,Dataset(10),0,'rain')

def test_counter_reset_rejected():
 with pytest.raises(ValueError,match='counter decreased'):
  accumulated_difference(Dataset(1),0,Dataset(5),0,'rain')

def test_small_roundoff_is_clipped():
 start,end=Dataset(10),Dataset(9.9999)
 assert np.all(accumulated_difference(end,0,start,0,'rain')==0)

def test_moving_nest_rejected():
 with pytest.raises(ValueError,match='Moving-nest'):
  assert_same_geography(Dataset(3,lat=30.01),0,Dataset(1),0)
 assert safe_accumulation(Dataset(5,lat=30.01),0,Dataset(1),0,'rain',label='unit test') is None

def test_dateline_wrap_equivalence():
 assert_same_geography(Dataset(5,lon=-179.999),0,Dataset(1,lon=180.001),0)
