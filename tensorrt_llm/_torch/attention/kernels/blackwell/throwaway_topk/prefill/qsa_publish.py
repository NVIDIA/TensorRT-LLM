# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cutlass
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

@dsl_user_op
def publish(ptr,score,tau,active,index,cursor,capacity,dtype,finite,skip_empty=False,packed=False,*,loc=None,ip=None):
    # Pointer is the beginning of one private warp bank. Capacity is in entries.
    nan='' if finite else 'setp.nan.f32 pn,$2,$2; or.pred ph,ph,pn; '
    cast=('cvt.rn.f16.f32 h,$2; cvt.f32.f16 rounded,h; mov.b32 value,rounded; '
          if dtype in (cutlass.Float16,cutlass.BFloat16) else 'mov.b32 value,$2; ')
    skip=('mov.b32 $0,$6; setp.eq.u32 pe,mask,0; @pe bra PUBLISH_DONE; ' if skip_empty else '')
    end=('PUBLISH_DONE: ' if skip_empty else '')
    asm=('{ .reg .pred ph,pa,pn,ps,pe; .reg .b32 mask,lower,prefix,total,pos,value; '
         '.reg .b64 address; .reg .b16 h; .reg .b32 packed_half; .reg .f32 rounded; '
         'setp.ge.f32 ph,$2,$3; '+nan+
         'setp.ne.u32 pa,$4,0; and.pred ph,ph,pa; '
         'vote.ballot.sync.b32 mask,ph,0xffffffff; '+skip+
         'mov.u32 lower,%lanemask_lt; and.b32 lower,lower,mask; '
         'popc.b32 prefix,lower; popc.b32 total,mask; '
         'add.u32 pos,$6,prefix; add.u32 $0,$6,total; '
         'setp.lt.u32 ps,pos,$7; and.pred ps,ps,ph; '
         'mad.wide.u32 address,pos,8,$1; '+cast+
         '@ps st.global.v2.b32 [address],{$5,value}; '+end+'}')
    if packed:
        assert dtype in (cutlass.Float16,cutlass.BFloat16)
        asm=asm.replace('mad.wide.u32 address,pos,8,$1;', 'mad.wide.u32 address,pos,4,$1;')
        asm=asm.replace(cast, 'cvt.rn.f16x2.f32 packed_half,$2,$2; prmt.b32 value,$5,packed_half,0x5410; ')
        asm=asm.replace('@ps st.global.v2.b32 [address],{$5,value};', '@ps st.global.b32 [address],value;')
    if dtype==cutlass.BFloat16:asm=asm.replace('f16','bf16')
    args=[ptr.toint(loc=loc,ip=ip).ir_value(loc=loc,ip=ip),
          cutlass.Float32(score).ir_value(loc=loc,ip=ip),cutlass.Float32(tau).ir_value(loc=loc,ip=ip),
          cutlass.Int32(active).ir_value(loc=loc,ip=ip),cutlass.Int32(index).ir_value(loc=loc,ip=ip),
          cutlass.Int32(cursor).ir_value(loc=loc,ip=ip),cutlass.Int32(capacity).ir_value(loc=loc,ip=ip)]
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type,args,asm,'=r,l,f,f,r,r,r,r,~{memory}',has_side_effects=True,loc=loc,ip=ip))

@dsl_user_op
def publish_pair(ptr0,ptr1,score0,score1,tau0,tau1,active0,active1,index,cursor0,cursor1,capacity,dtype,skip_empty,packed=False,ordered_threshold=False,bounds=None,*,loc=None,ip=None):
    from cutlass._mlir import ir
    cast=('cvt.rn.f16.f32 h0,$4; cvt.f32.f16 rounded0,h0; mov.b32 value0,rounded0; '
          'cvt.rn.f16.f32 h1,$5; cvt.f32.f16 rounded1,h1; mov.b32 value1,rounded1; '
          if dtype in (cutlass.Float16,cutlass.BFloat16) else 'mov.b32 value0,$4; mov.b32 value1,$5; ')
    skip=('mov.b32 $0,$11; mov.b32 $1,$12; or.b32 mask,mask0,mask1; '
          'setp.eq.u32 pe,mask,0; @pe bra PAIR_DONE; ' if skip_empty else '')
    asm=('{ .reg .pred p0,p1,pn,pa,ps0,ps1,pe; '
         '.reg .b32 mask,mask0,mask1,lower,lower0,lower1,prefix0,prefix1,total0,total1,pos0,pos1,value0,value1; '
         '.reg .b64 address0,address1; .reg .b16 h0,h1; .reg .b32 packed_half0,packed_half1; .reg .f32 rounded0,rounded1; '
         'setp.ge.f32 p0,$4,$6; setp.nan.f32 pn,$4,$4; or.pred p0,p0,pn; '
         'setp.ge.f32 p1,$5,$7; setp.nan.f32 pn,$5,$5; or.pred p1,p1,pn; '
         'setp.ne.u32 pa,$8,0; and.pred p0,p0,pa; '
         'setp.ne.u32 pa,$9,0; and.pred p1,p1,pa; '
         'vote.ballot.sync.b32 mask0,p0,0xffffffff; vote.ballot.sync.b32 mask1,p1,0xffffffff; '+skip+
         'mov.u32 lower,%lanemask_lt; and.b32 lower0,lower,mask0; and.b32 lower1,lower,mask1; '
         'popc.b32 prefix0,lower0; popc.b32 prefix1,lower1; '
         'popc.b32 total0,mask0; popc.b32 total1,mask1; '
         'add.u32 pos0,$11,prefix0; add.u32 pos1,$12,prefix1; '
         'add.u32 $0,$11,total0; add.u32 $1,$12,total1; '
         'setp.lt.u32 ps0,pos0,$13; and.pred ps0,ps0,p0; '
         'setp.lt.u32 ps1,pos1,$13; and.pred ps1,ps1,p1; '
         'mad.wide.u32 address0,pos0,8,$2; mad.wide.u32 address1,pos1,8,$3; '+cast+
         '@ps0 st.global.v2.b32 [address0],{$10,value0}; '
         '@ps1 st.global.v2.b32 [address1],{$10,value1}; '+('PAIR_DONE: ' if skip_empty else '')+'}')
    active_predicates=('setp.ne.u32 pa,$8,0; and.pred p0,p0,pa; '
                           'setp.ne.u32 pa,$9,0; and.pred p1,p1,pa; ')
    if bounds is not None:
        asm=asm.replace(active_predicates,
            'setp.ge.and.s32 p0,$10,$14,p0; setp.lt.and.s32 p0,$10,$15,p0; '
            'setp.ge.and.s32 p1,$10,$16,p1; setp.lt.and.s32 p1,$10,$17,p1; ')
    if ordered_threshold:
        # The caller canonicalizes a NaN filtering bound to +inf once per row.
        asm=asm.replace('setp.ge.f32 p0,$4,$6; setp.nan.f32 pn,$4,$4; or.pred p0,p0,pn;',
                        'setp.geu.f32 p0,$4,$6;')
        asm=asm.replace('setp.ge.f32 p1,$5,$7; setp.nan.f32 pn,$5,$5; or.pred p1,p1,pn;',
                        'setp.geu.f32 p1,$5,$7;')
    if packed:
        assert dtype in (cutlass.Float16,cutlass.BFloat16)
        asm=asm.replace('mad.wide.u32 address0,pos0,8,$2;', 'mad.wide.u32 address0,pos0,4,$2;')
        asm=asm.replace('mad.wide.u32 address1,pos1,8,$3;', 'mad.wide.u32 address1,pos1,4,$3;')
        asm=asm.replace(cast,
            'cvt.rn.f16x2.f32 packed_half0,$5,$4; prmt.b32 value0,$10,packed_half0,0x5410; '
            'prmt.b32 value1,$10,packed_half0,0x7610; ')
        asm=asm.replace('@ps0 st.global.v2.b32 [address0],{$10,value0};', '@ps0 st.global.b32 [address0],value0;')
        asm=asm.replace('@ps1 st.global.v2.b32 [address1],{$10,value1};', '@ps1 st.global.b32 [address1],value1;')
    if dtype==cutlass.BFloat16:asm=asm.replace('f16','bf16')
    args=[p.toint(loc=loc,ip=ip).ir_value(loc=loc,ip=ip) for p in (ptr0,ptr1)]
    args += [cutlass.Float32(x).ir_value(loc=loc,ip=ip) for x in (score0,score1,tau0,tau1)]
    args += [cutlass.Int32(x).ir_value(loc=loc,ip=ip) for x in (active0,active1,index,cursor0,cursor1,capacity)]
    if bounds is not None:
        args += [cutlass.Int32(x).ir_value(loc=loc,ip=ip) for x in bounds]
    result=llvm.inline_asm(ir.Type.parse('!llvm.struct<(i32,i32)>'),args,asm,
        '=r,=r,l,l,f,f,f,f,r,r,r,r,r,r'+(',r,r,r,r' if bounds is not None else '')+',~{memory}',has_side_effects=True,loc=loc,ip=ip)
    return tuple(cutlass.Int32(llvm.extractvalue(cutlass.Int32.mlir_type,result,[i],loc=loc,ip=ip)) for i in range(2))

@dsl_user_op
def pack_half_index(value,index,dtype=cutlass.Float16,*,loc=None,ip=None):
    asm="{ .reg .b16 h; .reg .b32 bits; cvt.rn.f16x2.f32 bits,$1,$1; prmt.b32 $0,$2,bits,0x5410; }"
    if dtype==cutlass.BFloat16:asm=asm.replace("f16x2","bf16x2")
    result=llvm.inline_asm(cutlass.Int32.mlir_type,
        [cutlass.Float32(value).ir_value(loc=loc,ip=ip),cutlass.Int32(index).ir_value(loc=loc,ip=ip)],
        asm,
        "=r,f,r",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Int32(result)
