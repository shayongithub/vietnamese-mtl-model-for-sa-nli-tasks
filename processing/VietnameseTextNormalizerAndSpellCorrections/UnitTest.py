# -*- coding: utf-8 -*-
from __future__ import annotations

import VietnameseTextNormalizer

a = VietnameseTextNormalizer.Normalize(
    'UCS2 : Tôi làm việ ở ban công ngệ FPT, '
    'tôi là người viêt nam. hôm nay tôi ko thích ăn mì tôm. tôi làm đc 2 bài tập.',
)
print(a)
print(type(a))


#   print ("Kì vọng  bằng: '1', huyền: '2', ngã: '3', hỏi: '4', sắc: '5', nặng: '6' ")
#   print ("bằng  : " + VietnameseTextNormalizer.GetFirstTone("tôi"))
#   print ("huyền : " + VietnameseTextNormalizer.GetFirstTone("huyền"))
#   print ("ngã   : " + VietnameseTextNormalizer.GetFirstTone("ngã"))
#   print ("hỏi   : " + VietnameseTextNormalizer.GetFirstTone("hỏi"))
#   print ("sắc   : " + VietnameseTextNormalizer.GetFirstTone("sắc"))
#   print ("nặng  : " + VietnameseTextNormalizer.GetFirstTone("nặng"))
#   print ("Other : " + VietnameseTextNormalizer.GetFirstTone(''))
#   print ("Other : " + VietnameseTextNormalizer.GetFirstTone(u''))
#
#
#   a=VietnameseTextNormalizer.GetIBase(u"UCS2 : Tôi làm việ ở ban công ngệ FPT, tôi là người viêt nam. hôm nay tôi ko thích ăn mì tôm. tôi làm đc 2 bài tập.");  # noqa: E501
#   print (a)
#   print (type(a))
#
#
#   b=VietnameseTextNormalizer.GetIBase(" UTF8 : Tôi làm việ ở ban công ngệ FPT, tôi là người viêt nam. hôm nay tôi ko thích ăn mì tôm. tôi làm đc 2 bài tập.");  # noqa: E501
#   print (b)
#   print (type(b))
