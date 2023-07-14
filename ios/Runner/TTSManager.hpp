//
//  TTSManager.hpp
//  Runner
//
//  Created by ductran on 14/07/2023.
//

#ifndef TTSManager_hpp
#define TTSManager_hpp

#include <stdio.h>
#include <vector>

std::vector<float> generateTTSAudio(const char* lightspeech_path, const char* mbmelgan_path);
#endif /* TTSManager_hpp */
