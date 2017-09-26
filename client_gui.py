# !/usr/bin/python
# Ok, I am very happy to coding
# author: David, Citybuster
import cv2
import socket
import wx
import os
import os.path
import multiprocessing as mp
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
os.system('rm -r ./img')
os.system('mkdir img')

# class for iimage sequence player
class MyFrame(wx.Frame):
    def __init__(self, parent, id, title):
        self.max_frame = 0
        wx.Frame.__init__(self, parent, id, title, size=(720, 500))
        #panel = wx.Panel(self, wx.ID_ANY)

        menubar = wx.MenuBar()
        file = wx.Menu()
        quit = wx.MenuItem(file, 105, '&Quit\tCtrl+Q', 'Quit the Application')
        file.AppendItem(quit)
        menubar.Append(file, '&File')

        #
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)

        # panel for screen and control
        self.pnl1 = wx.Panel(self, -1)
        self.picture = wx.StaticBitmap(self.pnl1, pos=(40,20))
        self.pnl1.SetBackgroundColour(wx.BLACK)
        pnl2 = wx.Panel(self, -1 )

        self.sld = wx.Slider(pnl2, -1, 1000, 1, 1000)
        self.pause = wx.Button(pnl2, wx.ID_ANY, 'start')
        self.play  = wx.Button(pnl2, wx.ID_ANY, 'stop')
        self.pause.Bind(wx.EVT_BUTTON, self.onStart)
        self.play.Bind(wx.EVT_BUTTON, self.onStop)

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        hbox1.Add(self.sld, 1)
        hbox2.Add(self.pause)
        hbox2.Add(self.play, flag=wx.RIGHT, border=5)
        hbox2.Add((150, -1), 1, flag=wx.EXPAND | wx.ALIGN_RIGHT)

        vbox.Add(hbox1, 1, wx.EXPAND | wx.BOTTOM, 10)
        vbox.Add(hbox2, 1, wx.EXPAND)
        pnl2.SetSizer(vbox)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.pnl1, 1, flag=wx.EXPAND)
        #sizer.Add(self.picture, 1, flag=wx.EXPAND)
        sizer.Add(pnl2, flag=wx.EXPAND | wx.BOTTOM | wx.TOP, border=10)

        self.SetMinSize((350, 300))
        self.SetSizer(sizer)
        self.SetMenuBar(menubar)
        self.Centre()
        self.Bind(wx.EVT_MENU, self.OnQuit, id=105)

    def OnQuit(self, event):

        print 'Shut down client and server'

        # Shut down the server
        s = socket.socket()
        s.connect(('10.117.18.17', 8888))
        for i in xrange(30):
            s.send('stop')
        s.close()
        # Shut down the client
        os.system('ps aux | grep client_gui.py | cut -c 10-14 | xargs kill -s 9')
        os.system('ps aux | grep python* | cut -c 10-14 | xargs kill -s 9')

        self.Close()

    def onStart(self, event):
        print "starting timer"
        self.timer.Start(60)

    def onStop(self, event):
        print "timer stop"
        self.timer.Stop()

    def update(self, event):
        frame_start = self.sld.GetValue()
        dir_img = dir_path + '/img'
        #print(os.listdir(dir_path))
        self.max_frame = len([name for name in os.listdir(dir_img) if os.path.isfile(os.path.join(dir_img, name))])
        #self.sld.SetMax(max_frame+3)
        #print(max_frame)
        #print(frame_start)
        self.max_frame -= 1
        print self.max_frame
        if frame_start == self.sld.GetMax():
            img_url = dir_img + '/%04d.jpg' % self.max_frame
        else:
            current_frame = self.max_frame-(self.sld.GetMax()-frame_start)
            if current_frame < 1:
                img_url = dir_img + '/%04d.jpg' % 1
                self.sld.SetValue(self.sld.GetMax()-self.max_frame+2)
            else:
                img_url = dir_img + '/%04d.jpg' % current_frame
        print(img_url)
        if os.path.isfile(img_url):
            print 'yes'
            self.picture.SetFocus()
            self.picture.SetBitmap(wx.Bitmap(img_url))
            if frame_start == self.sld.GetMax():
                self.sld.SetValue(frame_start)
            else:
                self.sld.SetValue(frame_start+1)


class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame(None, -1, 'Player')
        frame.Show(True)
        self.SetTopWindow(frame)
        return True


def gui():
    app = MyApp(0)
    app.MainLoop()


# this function can receive data from wti_graphic server, changed based on (http://stackoverflow.com/questions/9382045/send-a-file-through-sockets-in-python)
def client(save_queue):
    s = socket.socket()
    s.bind(('10.108.190.8', 1234))  # 10.108.198.32 is your client's IP address, you should change it for your convenience
    s.listen(10)
    frame_index = 1
    while True:
        sc, address = s.accept()

        print address
        frame_index += 1

        temp = sc.recv(1024)
        string_stream = temp
        while(temp):
            temp = sc.recv(1024)
            string_stream = string_stream + temp
        # data from server is string stream, which born form img(np.array) encode to string
        # if you want to display it, you should decode the data to be np.array for use
        img = cv2.imdecode(np.fromstring(string_stream, dtype=np.uint8), cv2.IMREAD_COLOR)
        save_queue.put(img)

        sc.close()

    s.close()

def save_img(save_queue):

    frame_index = 0

    while True:
        if not save_queue.empty():
            frame_index  += 1
            img_url = dir_path + '/img/%04d.jpg' % frame_index
            img = save_queue.get()
            cv2.imwrite(img_url, img)




if __name__ == '__main__':

    save_queue = mp.Queue()
    pclient = mp.Process(target=client, args=(save_queue,))
    pgui = mp.Process(target=gui)
    psave_img = mp.Process(target=save_img, args=(save_queue,))
    pclient.start()
    pgui.start()
    psave_img.start()
    pclient.join()
    pgui.join()
    psave_img.join()
