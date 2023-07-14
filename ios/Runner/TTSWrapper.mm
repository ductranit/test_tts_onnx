
#import "TTSWrapper.h"
#include "TTSManager.hpp" // Replace with your actual C++ file name

@implementation TTSClassWrapper

- (NSArray *)generateTTSAudioWithLightspeechPath:(NSString *)lightspeechPath mbmelganPath:(NSString *)mbmelganPath {
    std::vector<float> result = generateTTSAudio([lightspeechPath UTF8String], [mbmelganPath UTF8String]);

    NSMutableArray *resultArray = [NSMutableArray arrayWithCapacity:result.size()];
    for (int i = 0; i < result.size(); i++) {
        [resultArray addObject:@(result[i])];
    }

    return resultArray;
}

@end
