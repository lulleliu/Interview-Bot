# swagger_client.FurhatApi

All URIs are relative to *http://localhost:54321*

Method | HTTP request | Description
------------- | ------------- | -------------
[**furhat_attend_post**](FurhatApi.md#furhat_attend_post) | **POST** /furhat/attend | Attend a user/location
[**furhat_face_post**](FurhatApi.md#furhat_face_post) | **POST** /furhat/face | Change the texture and/or model
[**furhat_gesture_post**](FurhatApi.md#furhat_gesture_post) | **POST** /furhat/gesture | Perform a gesture
[**furhat_gestures_get**](FurhatApi.md#furhat_gestures_get) | **GET** /furhat/gestures | Get all gestures
[**furhat_get**](FurhatApi.md#furhat_get) | **GET** /furhat | Test connection
[**furhat_led_post**](FurhatApi.md#furhat_led_post) | **POST** /furhat/led | Change the colour of the LED strip
[**furhat_listen_get**](FurhatApi.md#furhat_listen_get) | **GET** /furhat/listen | Make the robot listen, and get speech results
[**furhat_listen_stop_post**](FurhatApi.md#furhat_listen_stop_post) | **POST** /furhat/listen/stop | Make the robot stop listening
[**furhat_say_post**](FurhatApi.md#furhat_say_post) | **POST** /furhat/say | Make the robot speak
[**furhat_say_stop_post**](FurhatApi.md#furhat_say_stop_post) | **POST** /furhat/say/stop | Make the robot stop talking
[**furhat_users_get**](FurhatApi.md#furhat_users_get) | **GET** /furhat/users | Get current users
[**furhat_visibility_post**](FurhatApi.md#furhat_visibility_post) | **POST** /furhat/visibility | Fade in/out the face
[**furhat_voice_post**](FurhatApi.md#furhat_voice_post) | **POST** /furhat/voice | Set the voice of the robot
[**furhat_voices_get**](FurhatApi.md#furhat_voices_get) | **GET** /furhat/voices | Get all the voices on the robot


# **furhat_attend_post**
> Status furhat_attend_post(user=user, userid=userid, location=location)

Attend a user/location

Provides 3 modes of attention.   1. Attend user based on enum (closest, other, random)   2. Attend user based on it's id (can be retrieved by using /furhat/users)   3. Attend location based on coordinates (x,y,z)

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()
user = 'user_example' # str | Make Furhat attend a user (closest, other, random) (optional)
userid = 'userid_example' # str | Make Furhat attend a user specified by id (optional)
location = 'location_example' # str | Make Furhat attend location, usage: x,y,z. Example -20.0,-5.0,23.0 (optional)

try:
    # Attend a user/location
    api_response = api_instance.furhat_attend_post(user=user, userid=userid, location=location)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_attend_post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **user** | **str**| Make Furhat attend a user (closest, other, random) | [optional] 
 **userid** | **str**| Make Furhat attend a user specified by id | [optional] 
 **location** | **str**| Make Furhat attend location, usage: x,y,z. Example -20.0,-5.0,23.0 | [optional] 

### Return type

[**Status**](Status.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **furhat_face_post**
> Status furhat_face_post(model=model, texture=texture, mask=mask, character=character)

Change the texture and/or model

Changes the mask and character, or model and texture (legacy), based on the property name. Case sensitive. Names can be retrieved from the web interface

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()
model = 'model_example' # str | Change the model of the robot (legacy) (optional)
texture = 'texture_example' # str | Change the texture of the robot (legacy) (optional)
mask = 'mask_example' # str | Change the mask of the robot (optional)
character = 'character_example' # str | Change the character of the robot (optional)

try:
    # Change the texture and/or model
    api_response = api_instance.furhat_face_post(model=model, texture=texture, mask=mask, character=character)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_face_post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **model** | **str**| Change the model of the robot (legacy) | [optional] 
 **texture** | **str**| Change the texture of the robot (legacy) | [optional] 
 **mask** | **str**| Change the mask of the robot | [optional] 
 **character** | **str**| Change the character of the robot | [optional] 

### Return type

[**Status**](Status.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **furhat_gesture_post**
> Status furhat_gesture_post(name=name, blocking=blocking, definition=definition)

Perform a gesture

Performs a gesture based on   1. Gesture name (retrieve by GET request to /furhat/gestures)   2. Gesture definition, see example

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()
name = 'name_example' # str | The gesture to do (optional)
blocking = true # bool | Whether to wait until the gesture is performed (default false) (optional)
definition = swagger_client.GestureDefinition() # GestureDefinition |  (optional)

try:
    # Perform a gesture
    api_response = api_instance.furhat_gesture_post(name=name, blocking=blocking, definition=definition)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_gesture_post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **name** | **str**| The gesture to do | [optional] 
 **blocking** | **bool**| Whether to wait until the gesture is performed (default false) | [optional] 
 **definition** | [**GestureDefinition**](GestureDefinition.md)|  | [optional] 

### Return type

[**Status**](Status.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **furhat_gestures_get**
> list[Gesture] furhat_gestures_get()

Get all gestures

Returns a JSON array with all gestures on the system (names + duration).

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()

try:
    # Get all gestures
    api_response = api_instance.furhat_gestures_get()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_gestures_get: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**list[Gesture]**](Gesture.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **furhat_get**
> furhat_get()

Test connection

Used to verify if the server is running, return \"hello world\" upon success

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()

try:
    # Test connection
    api_instance.furhat_get()
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_get: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

void (empty response body)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **furhat_led_post**
> Status furhat_led_post(red=red, green=green, blue=blue)

Change the colour of the LED strip

Changes the LED strip of the robot, colours can be between 0-255 (above 255 is changed to 255). Any parameter not provided defaults to 0.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()
red = 56 # int | The amount of red (optional)
green = 56 # int | The amount of green (optional)
blue = 56 # int | The amount of blue (optional)

try:
    # Change the colour of the LED strip
    api_response = api_instance.furhat_led_post(red=red, green=green, blue=blue)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_led_post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **red** | **int**| The amount of red | [optional] 
 **green** | **int**| The amount of green | [optional] 
 **blue** | **int**| The amount of blue | [optional] 

### Return type

[**Status**](Status.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **furhat_listen_get**
> Status furhat_listen_get(language=language)

Make the robot listen, and get speech results

Blocking call to get user speech input, language defaults to english_US. Language parameter can be used to provide a different language. Return values can be found in the Status object as message and can be:   - User speech   - SILENCE   - INTERRUPTED   - FAILED 

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()
language = 'language_example' # str | The language to listen for, defaults to en-US (optional)

try:
    # Make the robot listen, and get speech results
    api_response = api_instance.furhat_listen_get(language=language)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_listen_get: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **language** | **str**| The language to listen for, defaults to en-US | [optional] 

### Return type

[**Status**](Status.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **furhat_listen_stop_post**
> Status furhat_listen_stop_post()

Make the robot stop listening

Aborts the listen

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()

try:
    # Make the robot stop listening
    api_response = api_instance.furhat_listen_stop_post()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_listen_stop_post: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**Status**](Status.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **furhat_say_post**
> Status furhat_say_post(text=text, url=url, blocking=blocking, lipsync=lipsync, abort=abort)

Make the robot speak

Makes the robot speak by either using text, or a URL (linking to a.wav file). If generatelipsync=true, it uses a .pho file hosted on the same url, or generates phonemes by itself.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()
text = 'text_example' # str | A string depicting a utterance the robot should say. (optional)
url = 'url_example' # str | A url link to a audio file (.wav) (optional)
blocking = true # bool | Whether to wait until the speech is performed (default false) (optional)
lipsync = true # bool | If a URL is provided, indicate if lipsync files should be generated/looked for. (optional)
abort = true # bool | Stops the current speech of the robot. (optional)

try:
    # Make the robot speak
    api_response = api_instance.furhat_say_post(text=text, url=url, blocking=blocking, lipsync=lipsync, abort=abort)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_say_post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **text** | **str**| A string depicting a utterance the robot should say. | [optional] 
 **url** | **str**| A url link to a audio file (.wav) | [optional] 
 **blocking** | **bool**| Whether to wait until the speech is performed (default false) | [optional] 
 **lipsync** | **bool**| If a URL is provided, indicate if lipsync files should be generated/looked for. | [optional] 
 **abort** | **bool**| Stops the current speech of the robot. | [optional] 

### Return type

[**Status**](Status.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **furhat_say_stop_post**
> Status furhat_say_stop_post()

Make the robot stop talking

Stops the current speech.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()

try:
    # Make the robot stop talking
    api_response = api_instance.furhat_say_stop_post()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_say_stop_post: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**Status**](Status.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **furhat_users_get**
> list[User] furhat_users_get()

Get current users

Get all current users (max: 2). Returns a JSON array containg Users (Rotation, Location, id).

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()

try:
    # Get current users
    api_response = api_instance.furhat_users_get()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_users_get: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**list[User]**](User.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **furhat_visibility_post**
> Status furhat_visibility_post(visible, duration=duration)

Fade in/out the face

Triggers an animation which fades the face out to black, or in again, with a set duration in the range of (1,10000] ms. Invalid or missing durations will be set to the default value of 2000 ms. This command is only applicable to the FaceCore face engine.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()
visible = true # bool | Whether the face should be made visible or not
duration = 56 # int | Duration of the fade animation in milliseconds (optional)

try:
    # Fade in/out the face
    api_response = api_instance.furhat_visibility_post(visible, duration=duration)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_visibility_post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **visible** | **bool**| Whether the face should be made visible or not | 
 **duration** | **int**| Duration of the fade animation in milliseconds | [optional] 

### Return type

[**Status**](Status.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **furhat_voice_post**
> Status furhat_voice_post(name)

Set the voice of the robot

Sets the voice of the robot using the name of the voice, can be requested by doing a GET request on this endpoint.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()
name = 'name_example' # str | The name of the voice

try:
    # Set the voice of the robot
    api_response = api_instance.furhat_voice_post(name)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_voice_post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **name** | **str**| The name of the voice | 

### Return type

[**Status**](Status.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **furhat_voices_get**
> list[Voice] furhat_voices_get()

Get all the voices on the robot

Returns a JSON array with voice names + languages.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.FurhatApi()

try:
    # Get all the voices on the robot
    api_response = api_instance.furhat_voices_get()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling FurhatApi->furhat_voices_get: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**list[Voice]**](Voice.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

