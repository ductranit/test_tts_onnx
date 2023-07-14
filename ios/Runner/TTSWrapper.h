//
//  TTSWrapper.h
//  Runner
//
//  Created by ductran on 14/07/2023.
//

#ifndef TTSWrapper_h
#define TTSWrapper_h

// Wrapper.h
#import <Foundation/Foundation.h>

@interface TTSClassWrapper : NSObject

- (NSArray *)generateTTSAudioWithLightspeechPath:(NSString *)lightspeechPath mbmelganPath:(NSString *)mbmelganPath;

@end

#endif /* TTSWrapper_h */
