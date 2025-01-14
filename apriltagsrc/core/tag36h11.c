/* (C) 2013-2015, The Regents of The University of Michigan
All rights reserved.

This software may be available under alternative licensing
terms. Contact Edwin Olson, ebolson@umich.edu, for more information.

   An unlimited license is granted to use, adapt, modify, or embed the 2D
barcodes into any medium.

   Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
 */

#include <stdlib.h>
#include "apriltag.h"

apriltag_family_t *tag36h11_create()
{
   apriltag_family_t *tf = calloc(1, sizeof(apriltag_family_t));
   tf->name = strdup("tag36h11");
   tf->black_border = 1;
   tf->d = 6;
   tf->h = 11;
   tf->ncodes = 22;
   tf->codes = calloc(22, sizeof(uint64_t));
   tf->codes[0] = 0x0000000d97f18b49UL;
   tf->codes[1] = 0x0000000dd280910eUL;
   tf->codes[2] = 0x0000000e479e9c98UL;
   tf->codes[3] = 0x0000000ebcbca822UL;
   tf->codes[4] = 0x0000000f31dab3acUL;
   tf->codes[5] = 0x0000000056a5d085UL;
   tf->codes[6] = 0x000000010652e1d4UL;
   tf->codes[7] = 0x000000022b1dfeadUL;
   tf->codes[8] = 0x0000000265ad0472UL;
   tf->codes[9] = 0x000000034fe91b86UL;
   tf->codes[10] = 0x00000003ff962cd5UL;
   tf->codes[11] = 0x000000043a25329aUL;
   tf->codes[12] = 0x0000000474b4385fUL;
   tf->codes[13] = 0x00000004e9d243e9UL;
   tf->codes[14] = 0x00000005246149aeUL;
   tf->codes[15] = 0x00000005997f5538UL;
   tf->codes[16] = 0x0000000683bb6c4cUL;
   tf->codes[17] = 0x00000006be4a7211UL;
   tf->codes[18] = 0x00000007e3158eeaUL;
   tf->codes[19] = 0x000000081da494afUL;
   tf->codes[20] = 0x0000000858339a74UL;
   tf->codes[21] = 0x00000008cd51a5feUL;
   return tf;
}

void tag36h11_destroy(apriltag_family_t *tf)
{
   free(tf->name);
   free(tf->codes);
   free(tf);
}

