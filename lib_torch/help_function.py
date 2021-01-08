def show_on_jupyter(img,color= None,title=None):
    import matplotlib.pyplot as plt
    import cv2
    """Show img on jupyter notebook. No return Value
    
    You should check the img's color space.
    I just consider about RGB color space & 1 ch color space(like green ch, gray space, ...)
    
    using matplotlib
    
    Parameters
    ----------
    img : 2-D Array
        numpy 2-D array
        opencv / sklearn / plt are avaliable.
        float / uint8 data type.
        
    color : string
        'gray' or 'None'
        'gray' means that img has a 1 ch.
        'None' means that img has a RGB ch.
        (default: None)
        
    title : string
        decide img's title
        (default : None)
        
    Returns
    -------
        No return value.
    
    Example
    -------
    >>> img = cv2.imread(img_path)
    >>> show_on_jupyter(img)
    
    img has a 1 ch
    >>> img = cv2.imread(img_path)
    >>> show_on_jupyter(img,'gray')
    """
    if color == 'gray':
        plt.axis("off")
        plt.title(title)
        plt.axis('off')
        plt.figure(figsize=(10, 10))
        plt.imshow(img,cmap=color)
        plt.show()
    elif color == None:
        plt.axis("off")
        plt.title(title)
        plt.axis('off')
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        print("Gray or None")
        
def show_histogram(img,color = None, dtype = 'int',background_thr = 1 ,background_checker = False):
    import cv2
    import matplotlib.pyplot as plt
    """Show histogram on jupyter notebook
    
    I consider about img's color space and data type.
    
    color space : RGB / 1ch color space(also gray)
    data type : uint8 or float64
    
    so, you should check your img's color space and data type.
    
    Parameters
    ----------
    img : 2-D Array
        numpy 2-D array
        opencv / sklearn / plt are avaliable.
        float / uint8 data type.
        
    color : string
        'gray' or 'None'
        'gray' means that img has a 1 ch.
        'None' means that img has a RGB ch.
        (default: None)
        
    dtype :  string
        'int' or 'float'
        img's data type
        float : [min,max] and divided 256 values
        int = [0,256]
        (default : int)
        
    Returns
    -------
        No return value.
    
    Example
    -------
    >>> img = cv2.imread(img_path)
    >>> show_histogram(img)
    
    img has a 1 ch
    >>> img = cv2.imread(img_path)
    >>> show_histogram(img,'gray')
    
    1ch img & float img
    >>> img = cv2.imread(img_path)
    >>> show_histogram(img,'gray','float')
    """
    if background_checker == False:
        if (color == None) and (dtype =='int'):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.show()
        if (color == None) and (dtype =='float'):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0,1])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.show()
        elif (color == 'gray')and (dtype=='float') :
            plt.hist(img.ravel(),256,[img.min(),img.max()])
            plt.title('Histogram for gray scale picture')
            plt.show()
        elif (color == 'gray') and (dtype == 'int'):
            plt.hist(img.ravel(),256,[0,256])
            plt.title('Histogram for gray scale picture')
            plt.show()
        else:
            print('check your parameter')
    else:
        if (color == None) and (dtype =='int'):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[background_thr,256])
                plt.plot(histr,color = col)
                plt.xlim([background_thr,256])
            plt.show()
        if (color == None) and (dtype =='float'):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0,1])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.show()
        elif (color == 'gray')and (dtype=='float') :
            plt.hist(img.ravel(),256,[img.min(),img.max()])
            plt.title('Histogram for gray scale picture')
            plt.show()
        elif (color == 'gray') and (dtype == 'int'):
            plt.hist(img.ravel(),256,[background_thr,256])
            plt.title('Histogram for gray scale picture')
            plt.show()
        else:
            print('check your parameter')