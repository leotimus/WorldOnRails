def create_and_save_saliency(image_agent, video_saliency, video_original, info):
    saliency = image_agent.score_frame(info, density=10, radius=10)
    cv2.imwrite(f'experiments/saliency_{get_time_mils()}.png', saliency)
    orig_img = info.wide_rgb.copy()
    saliency_img = image_agent.apply_saliency(saliency, info.wide_rgb, channel=0)
    video_saliency.write(saliency_img)
    video_original.write(orig_img)

    def saveSaliencyVideo(self, Ls):
        try:
            print('log videos....')
            start = time.time()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            tz = pytz.timezone('Europe/Berlin')
            time_stamp = str(datetime.now(tz))
            video_saliency = cv2.VideoWriter(
                f'experiments/saliency_throttle_{get_time_mils()}_video_{time_stamp}.avi', fourcc, 1, (480, 240))
            video_original = cv2.VideoWriter(
                f'experiments/original_throttle_{get_time_mils()}_video_{time_stamp}.avi', fourcc, 2, (480, 240))
            # torch.save(self.Ls, f'expirements/flush_{int(round(time.time() * 1000))}.data')

            pool = ThreadPool(processes=2)
            pool.starmap(create_and_save_saliency, zip(repeat(self), repeat(video_saliency), repeat(video_original),  Ls))
            pool.close()
            pool.terminate()
            pool.join()
            """
            for s in Ls:
                create_and_save_saliency(self, video, s)
            """
            tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
            print('\ttime: {}'.format(tstr), end='\r')
            cv2.destroyAllWindows()
            video_saliency.release()
            video_original.release()
            Ls.clear()
        except:
           traceback.print_exc()